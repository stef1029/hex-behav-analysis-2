# The trials table, column by column

`load_sessions(cohort)` returns one pandas DataFrame with one row per trial and every
session stacked. This page lists every column it can have, in the order they arrive.

```python
from hba2 import Cohort, load_sessions, describe_trials
trials = load_sessions(Cohort(path, batch="2026-10_v1"))
describe_trials(trials)       # what *this* table has, with example values
```

The columns come from three places, joined in this order:

1. **[The session's facts](#1-the-session)**, read from each NWB file.
2. **[The trial](#2-the-trial)**, **[the laser](#3-the-laser)** and
   **[the head](#4-the-head)**: the NWB file's trials table, written by `build_nwb`.
3. **[`sessions.csv`](#5-sessionscsv-generated-columns)**: its generated `_` columns, then
   **[your own labels](#6-sessionscsv-your-labels)**.

## Before you read the tables

**Times.** Every `t_*`, `start_time` and `stop_time` is in seconds on the session's
DAQ clock. Two sessions' times cannot be compared, and `session_start` is wall-clock
time, not a DAQ time ([why](principles.md#there-is-one-clock-and-it-is-the-daqs)).

**How "missing" is written.** It depends on the column's type:

| type | missing is |
|---|---|
| float | `NaN` |
| int (the port columns and the counts) | `-1` for a port, `0` for a count. These columns always have an answer |
| text | `""`, an empty string, not `NaN` |

**Not every table has every column.**
- **The NWB batch.** A file has the columns its `build_nwb` wrote, so a batch built
  before a column existed does not have it. `head_speed_px_s` and the
  `turn_rate_deg_s` columns are the usual ones missing.
- **`sessions.csv`.** It has the columns its last `build_index` wrote, so an index
  that has not been rebuilt lacks the newer ones, such as the cue, catch and
  stillness columns.
- **When a name is taken twice.** If a `sessions.csv` column has the same name as a
  column the file already gave, it is left out, so the file's version wins.

`describe_trials(trials)` tells you what is in front of you.

## 1. The session

From the session facts stored in each NWB file. The value is the same on every trial of
a session.

| column | type | meaning |
|---|---|---|
| `session_id` | text | `<YYMMDD>_<HHMMSS>_<mouse>`. The key every merge joins on |
| `session_start` | text | when the session started, ISO 8601 (`2026-05-27T15:38:13`), wall-clock time |
| `phase` | text | the protocol phase, as text (`"9"`). `_phase` below is the same value as a number |
| `protocol` | text | the protocol's name (`Autotraining (Visual)`) |
| `rig` | int (or text) | the rig number. It is the rig's name where the metadata gave no number |
| `mouse_id` | text | the mouse (`E4`) |

## 2. The trial

From post-processing's `trials.csv` and the rig's own log.

| column | type | meaning |
|---|---|---|
| `trial` | int | 0-based index of the trial within the session. Use this, with `session_id`, to address a trial |
| `start_time` | float | when the trial marker went high (s, DAQ) |
| `stop_time` | float | when the trial ended (s, DAQ) |
| `t_response` | float | when the chosen port's beam broke (s, DAQ). NaN if it never did |
| `response_time` | float | `t_response - start_time` (s). NaN without a response |
| `tracker` | text | the named counter the trial was recorded on (`multiple_leds_6x`) |
| `sub_tracker` | text | the sub-tracker within it. `default` where there is only one |
| `block` | text | the block of parameters that was active, from `trials.csv`'s `stage` |
| `trial_number` | int | the trial's number within its tracker, as the rig counted it. Starts at 1 and is not unique within a session |
| `outcome` | text | `success`, `failure` or `timeout`, as the controller logged it |
| `correct_port` | int | the port the cue asked for, 0–5. `-1` if none (a catch trial) |
| `chosen_port` | int | the port the animal touched, 0–5. `-1` if it touched none |
| `correct` | float | 1 if it chose the cued port, 0 if it chose another. **NaN where not defined**: no cued port, or no choice (a timeout). This is a default reading; define your own where a trial type needs it |
| `timed_out` | int | 1 where no port was chosen, else 0 |
| `stimulus_log` | text | what the controller presented, as JSON: a list of `{t, port, modality, stage, …}` |
| `event_log` | text | what else the protocol noted during the trial, as JSON: entries of `{t, name, …}` with `t` relative to the marker. Read it with `attach_notes` ([Loading](loading.md#notes-what-the-protocol-said)) |
| `details_json` | text | anything else the protocol attached to the trial, as JSON. Empty on older protocols |

## 3. The laser

From the DAQ's laser channel, so this is what was delivered, not what the protocol
asked for. The protocol's settings are in the `_laser_*` columns below.

| column | type | meaning |
|---|---|---|
| `laser_stim` | int | 1 where a pulse belongs to the trial. "Belongs" means it fell between 1 s before the marker (`LASER_PRE_S`) and `stop_time`. The window starts early because protocols arm the laser in windows that start before the trial |
| `n_laser_pulses` | int | how many pulses belong to the trial |
| `t_laser_on` | float | the first of them (s, DAQ). NaN if none |
| `laser_offset_s` | float | `t_laser_on - start_time` (s), usually negative. NaN if none |

## 4. The head

From pose (DeepLabCut, then the head-angle stage). Unless the name starts with
`response_`, a column is read at **the video frame at the trial start**. Every one is NaN
for a session without pose.

**Where the head was, and where it pointed:**

| column | type | meaning |
|---|---|---|
| `head_x`, `head_y` | float | head position (px, image coordinates) |
| `bearing_deg` | float | which way the head faced. Degrees, 0 = image right, 90 = image up |
| `cue_angle_deg` | float | where the cued port was relative to straight ahead, in [-180, 180). 0 is straight ahead and **positive is the mouse's left** (anticlockwise). NaN where there was no cued port |
| `touched_angle_deg` | float | the same for the port it went on to touch. NaN with no touch |
| `touched_distance_px` | float | how far the touched port was from the head (px) |
| `angle_port0` … `angle_port5` | float | the same angle for each of the six ports, so any port's angle needs no per-frame data |

**Whether to trust the reading** (what `qc_trials` thresholds):

| column | type | meaning |
|---|---|---|
| `ear_likelihood` | float | DeepLabCut's confidence in the ears, 0–1 |
| `frame_lag_s` | float | that frame's time minus `start_time` (s). Normally under one frame period; larger means the video has a gap there |
| `head_speed_px_s` | float | how fast the head moved at that frame: the larger of the steps into and out of it (px/s). A label that jumped reads as a speed no mouse reaches |
| `turn_rate_deg_s` | float | how fast the bearing turned at that frame, as a magnitude (deg/s). A bearing that flipped direction reads as a turn no neck allows |

**At the response**, the frame showing the beam break:

| column | type | meaning |
|---|---|---|
| `response_frame` | float | that frame's index in the video. NaN without a response |
| `response_frame_lag_s` | float | its time minus `t_response` (s) |
| `response_distance_px` | float | head to the touched port at the moment of the beam break (px). A poke is a few tens of pixels; hundreds means the head was elsewhere |
| `response_ear_likelihood` | float | `ear_likelihood` at that frame |
| `response_head_speed_px_s` | float | `head_speed_px_s` at that frame |
| `response_turn_rate_deg_s` | float | `turn_rate_deg_s` at that frame |

**Then one column added by the loader:**

| column | type | meaning |
|---|---|---|
| `_nwb` | text | the NWB file the trial came from. `load_frames`, `load_events` and the rest read it, so they need nothing else |

## 5. `sessions.csv`: generated columns

Every `_` column of the cohort's index, joined on `session_id`. `build_index` rebuilds
these from the small files only: the rig's metadata JSON, `session.json` and
`trials.csv`. See [the label tables](cohort-folder.md#the-two-label-tables).

By default (`labels="index"`) they are read from the current `sessions.csv`. With
`labels="snapshot"` they are the copy each NWB file stored when it was built.

Each of these can be used to pick sessions: `load_sessions(cohort, video_usable=True)`.
The short names are in `ALIASES` in [`load/sessions.py`](../load/sessions.py).

**Who, when, what**, from the metadata JSON:

| column | type | meaning |
|---|---|---|
| `_present` | bool | the session folder is still on disk. Always true in a loaded table, since a missing folder has no NWB |
| `_mouse_id` | text | the mouse |
| `_date` | text | `YYYY-MM-DD` |
| `_time` | text | `HH:MM:SS` |
| `_rig` | int | the rig number |
| `_protocol` | text | the protocol's name |
| `_phase` | int | the protocol phase |
| `_mouse_weight` | float | the weight typed in at the start (g) |
| `_requested_trials` | int | how many trials the session was set to run |

**The laser, as the protocol armed it.** This is what the protocol intended; what was
delivered is `laser_stim`.

| column | type | meaning |
|---|---|---|
| `_laser_enabled` | bool | stimulation was switched on |
| `_laser_windows_ms` | text | the enabled windows relative to the marker, `start..end`, joined by `;` (`-500..-20`). Usually the column that names the experimental group. Empty without the laser |
| `_laser_hz` | float | pulse frequency (Hz). NaN without the laser |
| `_laser_fixed_pulse` | bool | a fixed pulse rather than a train. NaN without the laser |

**The cue, as the protocol set it:**

| column | type | meaning |
|---|---|---|
| `_cue_duration_ms` | float | how long the cue stayed on. **0 means until the animal chose.** Blank means the protocol had no cue duration (a training phase) |
| `_wait_duration_ms` | float | the protocol's wait duration |
| `_catch_type` | text | the kind of catch trial: `wait` or `led`. Empty with no catch trials |
| `_catch_brightness` | float | the catch trials' LED brightness |
| `_catch_wait_s` | float | the catch trials' wait (s) |

**What happened**, from post-processing. These are blank until it has run:

| column | type | meaning |
|---|---|---|
| `_is_processed` | bool | post-processing has run |
| `_n_trials` | float | rows in the processed `trials.csv` |
| `_duration_s` | float | session length on the DAQ clock (s) |
| `_trackers` | text | every `tracker` that ran, sorted and joined by `;` |
| `_trial_types` | text | every `sub_tracker`, the same way |
| `_blocks` | text | every `block`, the same way |
| `_n_frames` | float | frames the camera tracker logged |
| `_fps` | float | the camera's frame rate |
| `_n_dropped_frames` | float | frames the camera reported dropping |
| `_daq_reliability` | float | the percentage of DAQ messages received intact, as the DAQ reported it |

**Stillness on the scales:**

| column | type | meaning |
|---|---|---|
| `_has_stillness` | bool | the scales readings were scored for stillness |
| `_still_model` | text | the model's name |
| `_still_model_version` | text | its version |
| `_still_model_digest` | text | a hash of what the model does. **Filter on this one**: two sessions with the same digest were scored by the same function |
| `_still_threshold` | float | the threshold used |
| `_still_threshold_off_model` | bool | the threshold differs from the one the model recommends |

**Whether the video's timing can be trusted.** Video frames are put on the DAQ clock by
position, so a video missing frames misplaces every pose after the first gap.

| column | type | meaning |
|---|---|---|
| `_video_frames` | float | frames the video holds (DeepLabCut's row count, or else the file's header) |
| `_frames_lost` | float | `_n_frames - _video_frames` |
| `_video_complete` | bool | no frames lost. Blank where it could not be checked |
| `_video_repair` | text | what the video-timing repair decided: `repaired` (frame times are right again), `unexplained` or `rejected` (not to be trusted). Blank if it never looked |
| `_video_repair_check` | text | whether heads sat on ports at beam breaks after the repair: `pass`, `fail` or blank |
| `_video_usable` | bool | **the one to filter on**: `_video_complete`, or repaired with a passed check |

**What else exists:**

| column | type | meaning |
|---|---|---|
| `_has_video` | bool | there is a video file |
| `_dlc_models` | text | every DeepLabCut model with output for the session, joined by `;` |
| `_has_pose` | bool | a model's output went on to the head-angle stage |
| `_has_ephys` | bool | there is an ephys recording |
| `_layout` | text | how the session folder is laid out: `nested` (files in `raw/`) or `flat` |

## 6. `sessions.csv`: your labels

Every column of `sessions.csv` without a leading `_`, in the order you left them.

- **Columns every index starts with:** `include`, `exclude_reason` and `notes`.
- **Columns you add,** such as `stim_position` or `group`, follow the same rules.
- **An empty column** comes back as NaN floats, not `""`.
- **Picking sessions by a label** is a keyword: `load_sessions(cohort, include=True)`.

## Added after loading

None of these are made by `load_sessions`. They are listed so you can tell where a
column came from.

| from | columns |
|---|---|
| `qc_trials` | `keep` (1 or 0), `qc_failed` (the first rule the trial broke) |
| `attach_events(trials, name=…)` | `t_<name>`, `port_<name>` (`count` gives `n_<name>`) |
| `attach_notes(trials, name=note(…))` | `<name>_<field>`, `has_<name>`, `pc_offset_<name>` (on the PC clock) |

These are covered in [Loading data](loading.md).
