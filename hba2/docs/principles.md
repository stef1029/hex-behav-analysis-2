# Principles

The package follows a few rules. Each one exists because an earlier library broke it and
paid for that. When a choice anywhere in the code looks odd, one of these is usually the
reason.

## There is one clock, and it is the DAQ's

Every stream is put on the DAQ's timestamps, and always by **counting pulses**, never by
comparing two clocks. The camera sends the DAQ a pulse for each frame it exposes, so
frame *n* happened at the *n*-th camera pulse. The scales work the same way, and so does
the Open Ephys board, which receives one pulse per DAQ message.
[`postprocessing/utils/clock.py`](../postprocessing/utils/clock.py) is the only module
that derives a timestamp. If a time looks wrong, that is the one file to read.

One stream cannot be put on that clock: what the protocol itself noted as it ran
(`t.event(...)`, `t.stimulus(...)`). Those notes are timed by the PC from when the
software opened the trial, and the delay between that moment and the DAQ's trial marker
has never been measured. So their times are kept apart, in columns named `pc_offset_*`.
`start_time + pc_offset_laser` means nothing, and the name is chosen so that writing it
looks wrong.

## Synchronise first, interpret later

Post-processing only puts streams on the clock. It never decides what counts as a cue, a
response, a correct trial or a reaction time. Those depend on the trial type, so they
belong to the code that reads the files. The processed files therefore name events by
the hardware that made them (`led_on`, `sensor_touch`), never by their meaning.

The NWB files do carry a few default readings, because nearly every analysis wants them:

- `correct`: chose the cued port.
- `response_time`: from the trial marker to the chosen port's beam breaking.
- `laser_stim`: a laser pulse belongs to this trial.

These are written down in the column descriptions and can be overridden. A plot that
defines correctness differently computes its own column from the events.

## Compute once, store the answer

Synchronisation and head angles are stable, so they are computed once and written to
files. The choices people change their minds about are never baked in:

- confidence thresholds
- which trials to leave out
- how to bin angles

Those choices are filters you apply when you load. Changing them should never mean
reprocessing a cohort.

## Nothing is deleted; things are marked

`qc_trials` adds a `keep` column and keeps every row:

- Code that reads the trial before can still see a trial whose pose was unusable.
- "Did QC remove more laser trials than control trials" is one `groupby`.

The same holds everywhere: a missing value is NaN or blank, never 0, and a stream that
could not be placed says so rather than being quietly filled in.

## What varies is data, not code

Four cases, and where each one lives:

- **A new experimental condition** is a value in a column, or a column you type into
  `sessions.csv`.
- **A new channel wired to a spare DAQ line** becomes its own pulse file automatically.
  A channel's role follows from its name; anything unrecognised is treated as a pulse
  source rather than dropped.
- **A protocol's own notes** are passed through unread, because their vocabulary
  belongs to the protocol.
- **A new label** is a new column heading in a spreadsheet, with nothing to register
  anywhere.

## Your labels are never overwritten

`sessions.csv` and `neurons.csv` follow one rule:

- Columns starting with `_` belong to the file and are rebuilt every time.
- Every other column is yours and is never touched.

See [the cohort folder](cohort-folder.md#the-two-label-tables).

## No hardcoded paths

Every path derives from the cohort folder you pass in. Nothing comes from the
environment, a config file or a module constant. Moving a cohort means changing the one
line at the call site, and nothing stores the cohort's name, because the folder is the
name.

## Ports are 0-based, everywhere

`SENSOR0` is the sensor at port 0, so the number in a channel name *is* the port.
Sessions recorded before the rename used 1-based names. The DAQ reader detects that from
the names and translates them on read, so nothing downstream sees two conventions.

## Readable files, readable code

- **Outputs are plain text** wherever their size allows: CSV and JSON you can open.
- **An NWB file is always rebuilt from those files**, never patched. A suspect one is
  rebuilt rather than investigated.
- **The code is written for a scientist checking what a figure did.** Files read top to
  bottom in the order the data moves. Every module's docstring explains why it is built
  the way it is.

## Blocks, stages and trackers

The rig has two systems for describing trials:

- **Blocks** are named sets of protocol parameters, such as cue duration and which
  ports are enabled. The engine calls the current one its "stage".
- **Trackers and sub-trackers** are the counters a trial is recorded on. There are
  always two levels, with `default` where there is no sub-tracker. Several blocks may
  share one tracker.

A trial's block is `stage` when the protocol passed it and `tracker` otherwise. The
trials table carries that as `block`. What each block's parameters were is in the
session's `-blocks.json`, written by the rig. Sessions recorded before that file existed
have only the block's name.
