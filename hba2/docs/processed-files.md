# The processed files

What post-processing writes into `<session>/processed/`. Everything here is plain CSV or
JSON, and everything except `notes.csv` is in seconds on the DAQ clock, counted from the
start of the DAQ recording.

You rarely read these directly: the NWB files carry all of it, and `load_sessions` reads
the NWB files. They matter when you want to check a session by hand, or to build something
the NWB file leaves out.

Sizes are from a real two-hour session with 830 trials.

| file | rows | what |
|---|---|---|
| `session.json` | — | everything the rig recorded about the setup, kept whole |
| `events.csv` | ~25,000 | every behaviour-channel transition |
| `trials.csv` | 830 | trial windows, and what the rig itself recorded |
| `notes.csv` | varies | what the protocol noted as it ran — **on the PC clock** |
| `frame_times.csv` | ~240,000 | video frame → time |
| `scales.csv` | ~680,000 | platform weight → time |
| `pulses_<name>.csv` | ~5,800 for laser | one per pulse source, e.g. `pulses_laser.csv` |
| `ephys_sync.csv`, `.json` | ~7,000 | the ephys clock against the DAQ's; ephys sessions only |
| `video_repair.json`, `frame_times_unrepaired.csv` | — | written by the video-timing repair, where it ran |

## `session.json`

The rig's metadata, kept whole rather than summarised:

- mouse and weight
- the protocol, with every protocol parameter
- the rig
- the camera
- the laser
- the ephys block

To that, post-processing adds:

- the recording facts from the camera
- which channels carried what
- the blocks that ran, with their parameters
- `trial_windows`
- `clock.t0_wall`, which turns DAQ-relative times into wall-clock time

## `events.csv`

| column | meaning |
|---|---|
| `t` | seconds |
| `event` | `sensor_touch` / `sensor_release`, `led_on` / `led_off`, `led_dim_on` / `led_dim_off`, `valve_open` / `valve_close`, `link_high` / `link_low` |
| `channel` | the DAQ channel, e.g. `SENSOR3` |
| `index` | the port, 0-based, parsed from the channel name; `-1` where not indexed |
| `in_trial` | `1` inside a trial window, `0` between trials |
| `trial` | the trial it falls in, `-1` between trials |

**Events are named by hardware, never by meaning.** An LED is not necessarily a cue, and
a touch is not necessarily a response; [Loading data](loading.md#events) is where you
say which is which.

**Three facts that are encoded once, here:**

- **A touch is a falling edge.** The sensors are active-low, and `sensor_touch` already
  accounts for that.
- **A dimmed LED is one event, not a stream of PWM pulses.** An LED below full
  brightness is pulsed at about 1 kHz. A run of those pulses is written as one
  `led_dim_on` and one `led_dim_off`, so "the first `led_on`" is never a distractor's
  pulse.
- **Some lines are left out on purpose.**
  - The camera and scales lines are not here; they have their own files.
  - The trial-marker line (`DAQ_LINK1`) is here, as `link_high` / `link_low`.
  - Events between trials are kept, with `in_trial = 0`.

## `trials.csv`

Deliberately minimal. It holds the window, taken from the trial-marker line, and fields
copied unchanged from the rig's own trials CSV. It holds no outcome or correctness
judgement of this package's own.

| column | meaning |
|---|---|
| `trial` | 0-based index |
| `t_start`, `t_end` | the window, from the trial-marker line |
| `t_response` | when the chosen port's beam broke inside the trial; blank if it never did |
| `tracker`, `sub_tracker` | the counter the trial was recorded on; `default` where there is no sub-tracker |
| `outcome` | `success` / `failure` / `timeout`, as the rig judged it |
| `correct_port`, `chosen_port` | the port the rig cued and the one it credited, 0-based |
| `trial_number` | the rig's own numbering |
| `stage` | the block name, when the protocol passed it |
| `stimulus_log`, `event_log`, `details_json` | what the protocol recorded, as JSON, unread |

- **Why `t_response` is here, not computed later.** The rig has already decided which
  port was chosen, and the marker line can stay high for seconds after the response. So
  `t_response` puts the rig's own decision on the clock, rather than making a new
  judgement.
- **The trial's times come from the DAQ, not the rig.** The rig's own
  `time_since_start_s` is the controller's software clock and drifts against the DAQ;
  the times here are taken from the marker line instead.

## `notes.csv`

What the protocol annotated as it ran, one row per field, because which keys a protocol
uses is exactly what varies:

| column | meaning |
|---|---|
| `trial` | the trial it was noted in |
| `note` | the annotation's number within the trial, so its fields can be put back together |
| `kind` | `stimulus` (from `t.stimulus(...)`) or `event` (from `t.event(...)`) |
| `note_name` | the event's name, or the stimulus's modality, e.g. `laser`, `gate_open`, `visual` |
| `pc_offset_s` | seconds from when the **software** opened the trial, on the PC clock |
| `key`, `value` | one field of it, e.g. `power_mw`, `8.0`; values are JSON, so numbers and lists read back as such |

**There is no `t` column, and there never will be.** The PC clock cannot be mapped onto
the DAQ's ([Principles](principles.md#there-is-one-clock-and-it-is-the-daqs)).

- **What the offsets can be used for.** Ordering notes within a trial, and the interval
  between two notes on the same trial. Those share a clock and a zero.
- **What the fields are good for.** They are the only record of things like which laser
  slot fired and at what power.

## `frame_times.csv`

| column | meaning |
|---|---|
| `frame` | row in the video file, which is what DeepLabCut indexes by |
| `frame_id` | the camera's own frame ID |
| `t` | seconds; NaN where there was no camera pulse to match |

`frame` and `frame_id` differ whenever the camera dropped frames: video row *N* is the
*N*-th *delivered* frame. Indexing by video row makes pose alignment correct by
construction.

## `scales.csv`

`t`, `weight_g` and `reading`. When the rig scored stillness, the file also has
`p_still`, `occupied`, `state`, `still_for`, `baseline_g` and `still_threshold`.

- **The stillness columns are carried, not computed.** A protocol may have waited on
  them, so they record what the session did.
- **Which model scored them** is in `session.json`.
- **When there are no stillness columns.** They are absent, not empty, on a session
  recorded before stillness existed.

## `pulses_<name>.csv`

`t_on`, `t_off` and `duration`, one row per pulse.

- **Which channels get one.** Any channel that is not arena hardware or a timing line:
  the laser today, anything wired to a spare line tomorrow.
- **How each file is named.** After its channel, so listing the folder shows which
  trains a session has.

## `ephys_sync.csv` and `ephys_sync.json`

One row a second: `pulse`, `t_ephys`, `t_daq` and `n_messages`. An ephys time between
two rows maps onto the straight line between them.

- **A mapping, not a conversion.** Behaviour stays on the DAQ clock. Spikes are brought
  onto it when a curation is exported.
- **What the JSON records.** Where the pulses were read from, and how closely the two
  clocks agreed.
