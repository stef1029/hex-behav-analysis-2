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

Each file's trials table holds one row per trial. `load_sessions` reads it and adds the
session's facts and its `sessions.csv` columns. Its columns fall into four groups:

- **the trial:** times, ports, outcome and `correct`
- **the laser:** pulses from the DAQ
- **the head:** at the trial start and at the response
- **the session:** its facts and labels

[The trials table](trials-table.md) lists every column, with its type and how it shows
a missing value.
