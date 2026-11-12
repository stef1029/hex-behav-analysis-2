# The NWB files

One file per session holds everything a figure needs:

- the trials, with the head at each trial's start and response
- the head on every frame
- every hardware event
- the laser pulses
- the port coordinates the angles were measured from
- the scales
- the session's own facts and your labels for it
- with a spike sort, its good units

Nothing downstream has to open a CSV, a DAQ file or a video. They are built by
[`nwb/build_nwb.py`](../nwb/build_nwb.py) ([the pipeline](pipeline.md#4-nwb-files-one-per-session-in-a-batch))
and read by [`load_sessions`](loading.md).

The files are standard NWB 2, so other tools read them too: pynwb, pynapple, Neurosift.
[ephys/TOOLS.md](../ephys/TOOLS.md) reviews the options.

## Batches

```
<cohort>/nwb/<batch>/recipe.json        the parameters, the code version, and per session
                                        what it resolved to, or why it was not built
<session>/nwb/<batch>/<session>.nwb
<session>/nwb/<batch>/<session>.json    that file's build record and its check
```

- **A batch is one set of choices.** It records one DeepLabCut model, and the sort and
  curation when there are units. Building again under a new name leaves old batches
  untouched, so a figure made from one batch can always be traced to it.
- **Which batch is read.** `load_sessions` reads the cohort's newest batch unless you
  name one.
- **When a session is skipped.** A session is not built when nothing it was made from
  has changed: the build record fingerprints every input.

## What is inside

```
/general
├── session_id, session_start_time, subject (mouse, weight that day), devices/rig_N
└── scratch/
    ├── session_facts      everything post-processing recorded about the session
    ├── session_labels     this session's row of sessions.csv when the file was built
    ├── build_record       sources, fingerprints and counts
    └── neuron_labels      with units: these neurons' rows of neurons.csv

/intervals
├── trials                 one row per trial — the trials table below
├── epochs                 one row per block
└── pulses_laser           one row per laser pulse, and the trial it fell in

/processing/behaviour
├── Position/head_position          (n, 2) px, every frame
├── CompassDirection/head_bearing   (n,) degrees
├── port_angles                     (n, 6) degrees
├── ear_likelihood, head_speed, turn_rate, bearing_method
├── video_frame, camera_frame_id    back to the video and the camera
├── events                          one row per hardware transition (as events.csv)
├── notes                           what the protocol said, on the PC clock
└── ports                           the six calibrated port coordinates

/acquisition/scales                 platform weight, ~85 Hz

/units                              with --sort --curation: one row per good unit,
                                    spike_times on the DAQ clock, waveform, metrics,
                                    neuron_id
```

**Every per-frame series shares one timestamps array.** That gives two things:

- There is no second copy of the clock to drift.
- Frames the video is missing show up as jumps in time, so a line drawn against time
  breaks at the gap by itself.

**Left out on purpose:**

- **The raw DAQ channels.** The events table already holds every transition on them.
- **DeepLabCut's other body parts.**
- **The video.** It is referenced only by its path.

A plot that needs any of these means a rebuild with them added, which takes a few
seconds a session.

## The trials table

`load_sessions` returns this table, with one row per trial. The columns fall into four
groups.

**The trial**, from post-processing and the rig:

| column | meaning |
|---|---|
| `trial` | 0-based index within the session |
| `start_time`, `stop_time` | the trial-marker window (s, DAQ clock) |
| `t_response` | when the chosen port's beam broke; NaN if it never did |
| `response_time` | `t_response - start_time` |
| `tracker`, `sub_tracker` | the counter the trial was recorded on |
| `block` | the block of parameters that was active |
| `trial_number` | the rig's own count within its tracker |
| `outcome` | `success` / `failure` / `timeout`, as the controller logged it |
| `correct_port`, `chosen_port` | 0–5; `-1` for none |
| `correct` | 1 chose the cued port, 0 chose another, **NaN where not defined**: no cue, or no choice (timeouts). A default reading; define your own if a trial type needs it |
| `timed_out` | 1 where no port was chosen |
| `stimulus_log`, `event_log`, `details_json` | what the protocol recorded, as JSON, unread |

**The laser**, from the DAQ's pulses rather than the protocol's notes:

| column | meaning |
|---|---|
| `laser_stim` | 1 where a pulse belongs to the trial. "Belongs" means it fell between 1 s before the marker and the trial's end, because protocols arm the laser in windows that start before the trial |
| `n_laser_pulses` | how many |
| `t_laser_on` | the first one (s) |
| `laser_offset_s` | `t_laser_on - start_time`, normally negative |

**The head**, from pose. These columns are measured at the trial-start frame unless they
say `response_`:

| column | meaning |
|---|---|
| `head_x`, `head_y` | position (px) |
| `bearing_deg` | which way the head faced (degrees; 0 = image right, 90 = image up) |
| `cue_angle_deg` | where the cued port sat relative to straight ahead, in [-180, 180). 0 is straight ahead, **positive is the mouse's left** |
| `touched_angle_deg` | the same for the port it went on to touch |
| `angle_port0` … `angle_port5` | the same for every port |
| `touched_distance_px` | how far the touched port was from the head |
| `ear_likelihood` | DeepLabCut's confidence in the ears |
| `frame_lag_s` | that frame's time minus `start_time`. Normally under one frame period; larger means the video has a gap there |
| `head_speed_px_s`, `turn_rate_deg_s` | how fast the head moved and turned at that frame. A label that jumped reads as a speed no mouse reaches |
| `response_frame`, `response_frame_lag_s` | the frame showing the response, and how far off it is |
| `response_distance_px` | head to the touched port at the moment its beam broke. A poke is a few tens of pixels; hundreds means the head was elsewhere |
| `response_ear_likelihood`, `response_head_speed_px_s`, `response_turn_rate_deg_s` | the same checks at the response |

**The session**, attached to every one of its trials by `load_sessions`:

| column | meaning |
|---|---|
| `session_id`, `mouse_id`, `rig`, `protocol`, `phase`, `session_start` | the session's own facts |
| `_mouse_id`, `_date`, `_n_trials`, `_video_usable`, … | every `_` column of `sessions.csv` |
| `include`, `stim_position`, … | every column you added to `sessions.csv` |
| `_nwb` | the file the trial came from, so `load_frames` and friends need nothing else |

`describe_trials(trials)` lists what a loaded table has, with example values.
