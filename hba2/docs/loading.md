# Loading data

Everything in [`load/`](../load/__init__.py): from a cohort's NWB files to one pandas
table of trials, and the extras you attach to it. After this point there is nothing
package-specific left. It is a DataFrame, and you filter, group and plot it however you
like.

```python
from hba2 import Cohort, load_sessions, qc_trials

cohort = Cohort("/path/to/cohort", batch="2026-10_v1")
trials = load_sessions(cohort, include=True, stim_position="all")
trials = qc_trials(trials, min_ear_likelihood=0.6, max_abs_frame_lag_s=0.1)
```

Everything here imports from the top-level package, `from hba2 import ...`:

- `load_sessions`, `qc_trials`, `kept`
- `load_frames`, `load_ports`, `load_events`, `load_notes`
- `attach_events` with `first`, `last`, `nth`, `count`; `attach_notes` with `note`
- `load_units`, `load_spikes`, `bin_trials`, `lump_neurons`
- `describe_sessions`, `describe_trials`, `describe_events`, `describe_notes`
- `passes`, the rules on their own

## `load_sessions`: the trials

```python
cohort = Cohort(path, batch="2026-10_v1")     # batch=None: the newest NWB batch
trials = load_sessions(cohort,                 # a Cohort, never a path
                       where=None,             # a pandas query on sessions.csv, for an "or"
                       only=None,              # a tuple of session IDs
                       labels="index",         # or "snapshot": the labels stored in each file
                       log_fn=print,           # watch what it does
                       **filters)              # columns of sessions.csv; see "rules" below
```

- **The batch comes from the cohort.** `load_sessions` reads the batch its `Cohort`
  was made with, and refuses a path or a `batch=`. So which files a figure came from
  is set in one place, where the cohort is made.
- **Picking sessions.** Sessions are picked by the columns of `sessions.csv`, using the
  same rule vocabulary as QC. A column can be named by a short alias (`mouse`, `date`,
  `rig`, `protocol`, `laser_window`, `video_usable`, `cue_duration`, …), by its own name
  (`_mouse_id`), or without the underscore.
- **Mistakes are reported.** A name that is not a column raises, listing the ones that
  are. A value nobody has raises, listing the values there are.
- **What comes back.** One row per trial, every session stacked, unsorted and unfiltered.
  [The NWB files](nwb-files.md#the-trials-table) lists every column.

To see what there is to choose from:

- `describe_sessions(cohort)`: the columns of `sessions.csv`, with their values
- `describe_trials(trials)`: the columns of a loaded table

## `qc_trials`: marking, not removing

```python
trials = qc_trials(trials,
                   min_ear_likelihood=0.6,          # DeepLabCut was sure of the ears
                   max_abs_frame_lag_s=0.1,         # a video frame close to the trial start
                   max_response_distance_px=100,    # the head really was at the touched port
                   max_head_speed_px_s=2500)        # the label did not jump
```

```
qc_trials: 10,935 trials → 9,459 keep, 1,476 marked
  min_ear_likelihood=0.6            366
  max_abs_frame_lag_s=0.1           108
  max_response_distance_px=100    1,002
```

- **The table keeps its height.** You get the same table back with two columns added:
  - `keep`, which is 1 or 0
  - `qc_failed`, the first rule each trial broke
- **Drop the marked trials when you compute**, with `trials[trials["keep"] == 1]` or
  `kept(trials)`.
- **Leave them in the table.** A trial with bad pose is still real as context: it is
  still "the trial before". And checking whether QC hit one condition harder than
  another is then one `groupby`.
- **Run it once, at the top of the notebook.** The thresholds sit there where everyone
  can see them.

## Rules: one vocabulary for choosing rows

`load_sessions`, `qc_trials` and `load_units` all take the same keywords:

| keyword | keeps rows where the column is |
|---|---|
| `column=value` | equal to it; a list means any of them; `None` in the list means "or blank" |
| `not_column=value` | anything but that |
| `min_column=x`, `max_column=x` | at or above, at or below (text orders too: `min_date="2026-06-01"`) |
| `max_abs_column=x`, `min_abs_column=x` | the same on the size, sign ignored (a lag of −0.3 s is as bad as +0.3 s) |

**Blanks never pass a threshold**: a trial with no `ear_likelihood` fails
`min_ear_likelihood`. `passes(frame, rule, value)` applies one rule to any DataFrame.

## Events

The trials table says when each trial began and ended, not when the cue came on, because
which channel counts as a cue depends on the protocol. You say it here, once, where it
can be seen:

```python
from hba2 import attach_events, first, count

trials = attach_events(trials,
    cue      = first("led_on",       within=(0.5, 0)),     # from 0.5 s before the marker
    touch    = first("sensor_touch", after="cue"),
    reward   = first("valve_open",   after="touch"),
    touches  = count("sensor_touch"),
)
# adds t_cue, port_cue, t_touch, port_touch, t_reward, port_reward, n_touches
```

- **`after=` matters.** The touch is the first touch after the cue, not the first in the
  trial. An animal already on a sensor would otherwise score a response time of zero.
- **Pick names the table does not already have.** A name becomes `t_<name>` and
  `port_<name>`. Naming one `response` would replace the table's own `t_response`, the
  chosen port's beam break, without warning.
- **The other selectors.** `last`, `nth(event, n)`, `port=` and `channel=` narrow the
  search further.
- **Nothing found is NaN**, never zero.
- **The raw transitions.** `load_events(trials, before=, after=)` returns every
  transition around the trials, one row each.

## Notes: what the protocol said

```python
from hba2 import attach_notes, note

trials = attach_notes(trials, laser=note("laser", "slot", "power_mw", "window_ms"))
# adds laser_slot, laser_power_mw, laser_window_ms, has_laser, pc_offset_laser
```

- **Use it for fields only the protocol knew**, such as which laser slot fired and at
  what power.
- **Each protocol has its own vocabulary.** `describe_notes(trials)` shows what a cohort
  actually recorded.
- **`pc_offset_*` is on the PC clock** and cannot be added to a DAQ time. It is only good
  for ordering, or for intervals within one trial
  ([why](principles.md#there-is-one-clock-and-it-is-the-daqs)).

## Frames: the head through a trial

```python
frames = load_frames(trials.head(12), before=0.5, after=0.5, port_angles=False)
```

- **What comes back.** One row per video frame per trial: `session_id`, `trial`,
  `video_frame`, `t`, `t_from_start`, `head_x`, `head_y`, `bearing_deg`,
  `ear_likelihood`, `head_speed_px_s` and `turn_rate_deg_s`.
- **Why it is a separate call.** It is about a hundred times the data, so ask only for
  the trials you will draw.
- **Gaps are left as gaps.** Frames the video does not have are absent, so `t` can jump.
  A line drawn through it should break there.
- **The port positions.** `load_ports(trials)` gives the six ports in the same pixel
  coordinates.

## Units and spikes

These need an NWB batch built with a sort and a curation.

```python
from hba2 import Cohort, load_sessions, load_units, load_spikes, bin_trials

trials = load_sessions(Cohort(path, batch="2026-10_v1"))
units  = load_units(trials, cell_type="narrow")      # one row per unit per session
spikes = load_spikes(trials, before=0.5, after=1.0, align="start_time", units=units)
rates  = bin_trials(trials, units, until="t_response")   # every neuron x trial, binned
```

- **Units are a table**, so choosing them is pandas. Marking one out is `keep = 0`, not a
  deleted row.
- **Your neuron labels are joined on.** `neurons.csv` is matched to every unit by
  `neuron_id`, so a column you typed there is something to select or split by.
- **`neuron_id` is the same cell across sessions.** Group by it as you would group trials
  by mouse.
- **Each session's spikes stay on that session's clock.** Nothing strings sessions
  together into one timeline.
- **Merging double-counted cells.** `lump_neurons(spikes)` merges two units of one session
  that the matching found to be one cell.
- **Binning through the trial.** `bin_trials` uses either fixed bins around one moment,
  or a trial stretched between two (cue → port) so trials of different lengths line up.
  Each trial's baseline rate sits beside it.

The [`load/spikes.py`](../load/spikes.py) docstring covers the details.
