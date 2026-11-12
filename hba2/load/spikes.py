"""
Units and their spikes, through the trials you hand it.
=======================================================

The batch NWB files hold a curation's good units beside the behaviour, on one clock. Two
calls read them, both taking a :func:`~.sessions.load_sessions` table the way
:func:`~.sessions.load_frames` does::

    trials = load_sessions(Cohort(path, batch="2026-10_first_ephys"))
    units  = load_units(trials)                       # one row per unit per session
    spikes = load_spikes(trials, before=0.5, after=1.0, align="start_time")
    rates  = bin_trials(trials, units, until="t_response")   # every neuron x trial, binned

**Units are a table, not objects**, so choosing them is pandas: ``units.query("shank ==
3 and snr > 4")``, and marking one out is ``keep = 0`` — never a deleted row. Hand the
chosen rows to ``load_spikes(units=...)`` to read only theirs.

**Your labels come with them.** ``neurons.csv`` (see ``cohort/neuron_index.py``) is joined onto
every unit by ``neuron_id``, so a column you typed there — ``cell_type``, ``optotagged``
— is a column here, to select by (``load_units(trials, cell_type="narrow")``) or to
split a figure by.

**A neuron across sessions is a column.** ``neuron_id`` is the same for units the
cross-session matching found to be one cell, so a plotter groups by it exactly as it
groups trials by mouse. Each session's spikes stay on that session's clock (``t``,
``t_rel``); nothing here strings sessions together into one timeline, because a gap of
days is not a stretch of silence — rates divide by time each session observed.

Two units of one session may share a ``neuron_id``, where the matching decided the sorter
split one cell. :func:`lump_neurons` merges them, dropping any spike the other unit also
caught.

**Binned through the trial.** :func:`bin_trials` gives every neuron's rate in every bin of
every trial — fixed bins around one moment, or a trial stretched between two (cue → port)
so trials of different lengths line up by phase — with each trial's rate in the pre-cue
baseline beside it. It is read once and handed to as many ephys plotters as want it.
:func:`describe_neurons` is the neuron-level table those plotters split neurons by.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO

from ..cohort import batch_cohort
from .sessions import NWB_COLUMN, _needs, _say_nothing, _typed
from .rules import bare as _bare_rule, passes

#: Columns of the NWB units table that are not one value per unit.
_RAGGED = ("spike_times", "obs_intervals", "electrodes", "electrode_group",
           "waveform_mean")

#: Two units' spikes this close together are one spike caught twice (s).
COINCIDENT_S = 0.0005


# =====================================================================
# The units, and their spikes around a moment
# =====================================================================

def _unit_column(units, name, rule=""):
    """The column a keyword means: its own name, or a generated one without the ``_``."""
    for candidate in (name, f"_{name}"):
        if candidate in units.columns:
            return candidate
    raise KeyError(f"no column {name!r} on the units. Yours: "
                   f"{', '.join(c for c in units.columns if not c.startswith('_'))}. "
                   f"Generated: {', '.join(c for c in units.columns if c.startswith('_'))}")


def load_units(trials, *, labels="index", log_fn=None, **filters):
    """
    Every unit in the sessions behind ``trials``, one row per unit per session, with
    your neuron labels attached.

    Args:
        trials: rows of a :func:`load_sessions` table from a batch.
        labels: ``"index"`` joins the cohort's ``neurons.csv`` as it is now — the file
            you edit. ``"snapshot"`` uses the rows each NWB stored when it was built,
            for files that have left this machine. The same choice ``load_sessions``
            offers for session labels.
        log_fn: called with a line per step.
        **filters: which units, in the vocabulary ``load_sessions`` uses —
            ``cell_type="narrow"``, ``not_include=False``, ``min_snr=4``,
            ``shank=[2, 3]``. A name can be a unit column, one of your label columns,
            or a generated one without its underscore (``n_sessions`` means
            ``_n_sessions``).

    Returns:
        ``session_id``, ``unit_id`` (Phy's cluster ID), ``neuron_id``, ``shank``,
        ``depth_um``, ``n_spikes``, ``firing_rate_hz``, the quality metrics, ``keep``,
        ``obs_start`` / ``obs_stop`` (s, the stretch the probe recorded), the rest of
        the units table as ``build_nwb`` describes it, then every column of
        ``neurons.csv`` — yours by name, generated ones underscored. A session whose
        file has no units contributes no rows, and the log says so.
    """
    log = log_fn or _say_nothing
    _needs(trials, NWB_COLUMN, "load_units")

    # ---- 1. Each session's units table ----------------------------------
    tables, without, snapshots = [], [], {}
    for path, rows in trials.groupby(NWB_COLUMN, sort=False):
        with NWBHDF5IO(str(path), "r") as io:
            nwb = io.read()
            stored = nwb.units
            if stored is None or len(stored.id) == 0:
                without.append(rows["session_id"].iloc[0])
                continue
            table = pd.DataFrame({
                column.name: column.data[:] for column in stored.columns
                if column.name not in _RAGGED and not column.name.endswith("_index")})
            intervals = [stored.get_unit_obs_intervals(row) for row in range(len(stored.id))]
            table.insert(0, "unit_id", stored.id[:])
            table.insert(0, "session_id", nwb.session_id)
            table["obs_start"] = [float(i[0][0]) for i in intervals]
            table["obs_stop"] = [float(i[-1][1]) for i in intervals]
            table[NWB_COLUMN] = str(path)
            if "neuron_labels" in nwb.scratch:
                snapshots.update(json.loads(nwb.scratch["neuron_labels"].data))
        tables.append(table)

    if without:
        line = f"  {len(without)} session(s) have no units: {', '.join(without[:5])}"
        if len(without) > 5:
            line += " ..."
        log(line)
    if not tables:
        raise ValueError("no units in any of those sessions — was the batch built with "
                         "--sort and --curation?")
    units = pd.concat(tables, ignore_index=True)
    units["neuron_id"] = units["neuron_id"].replace("", np.nan)

    # ---- 2. Your neuron labels ------------------------------------------
    if labels == "index":
        # The cohort's neurons.csv, found from a batch file's place.
        nwb_path = Path(units[NWB_COLUMN].iloc[0])
        cohort = batch_cohort(nwb_path)
        if cohort is None:
            log(f"  {nwb_path.name} is not in a batch — no neuron labels joined")
            neurons = pd.DataFrame(columns=["neuron_id"])
        elif not (cohort / "neurons.csv").exists():
            log(f"  no neurons.csv at {cohort} — no neuron labels joined")
            neurons = pd.DataFrame(columns=["neuron_id"])
        else:
            neurons = pd.read_csv(cohort / "neurons.csv", dtype={"neuron_id": str})
    elif not snapshots:
        neurons = pd.DataFrame(columns=["neuron_id"])
    else:
        # The rows each file stored, typed the way the index reads.
        neurons = pd.DataFrame.from_dict(snapshots, orient="index").reset_index(drop=True)
        neurons = neurons.map(_typed).astype({"neuron_id": str})

    # Every column of the neuron index onto every unit of that neuron. A label of yours
    # that shares a name with a unit column is left off and said so: silently replacing
    # ``shank`` from the units table with a hand-typed one would be worse than either.
    if not neurons.empty:
        clashes = [c for c in neurons.columns if c != "neuron_id" and c in units.columns]
        if clashes:
            log(f"  neurons.csv column(s) {', '.join(clashes)} share a name with a unit "
                "column and are not joined — rename them in neurons.csv")
        joined = [c for c in neurons.columns if c not in clashes]
        units = units.merge(neurons[joined], on="neuron_id", how="left")
    log(f"  {len(units)} unit(s) from {len(tables)} session(s), "
        f"{units['neuron_id'].nunique()} named neuron(s)")

    # ---- 3. The units asked for -----------------------------------------
    # A keyword matching none says what the column holds.
    if filters:
        match = pd.Series(True, index=units.index)
        for rule, value in filters.items():
            here = passes(units, rule, value, column_for=_unit_column)
            if not here.any():
                column = _unit_column(units, _bare_rule(rule), rule)
                shown = ", ".join(map(str, units[column].dropna().unique()[:8]))
                raise ValueError(f"no units have {rule}={value!r}. {column} holds: {shown}")
            match &= here
        units = units[match].reset_index(drop=True)
        log(f"  {len(units)} unit(s) after "
            + ", ".join(f"{name}={value!r}" for name, value in filters.items()))
    return units


def load_spikes(trials, *, before=0.0, after=0.0, align="start_time", units=None,
                log_fn=None):
    """
    Spikes around a moment of each trial, one row per spike per trial.

    Args:
        trials: rows of a :func:`load_sessions` table — all of them, or a handful.
        before, after: seconds either side of the moment.
        align: the column holding the moment, on the DAQ clock: ``start_time`` (the
            trial marker), ``stop_time``, or anything ``attach_events`` added. A trial
            where it is NaN is left out and counted in the log.
        units: rows of :func:`load_units` to read; default (None) every unit. Rows with
            ``keep == 0`` are left out.
        log_fn: called with a line per file opened.

    Returns:
        ``session_id``, ``trial``, ``unit_id``, ``neuron_id``, ``t`` (DAQ clock, s) and
        ``t_rel`` (``t`` minus the moment). A spike in two trials' windows appears once
        per trial. A trial in which a unit did not fire has no rows for it, so a rate's
        denominator is the trials you passed in, not the trials found here;
        ``attrs`` holds ``align``, ``before`` and ``after`` for working it out.
    """
    log = log_fn or _say_nothing
    _needs(trials, NWB_COLUMN, "load_spikes")
    if align not in trials:
        raise KeyError(f"no column {align!r} to align to. Moments on the DAQ clock: "
                       "start_time, stop_time, or one attach_events added")
    if units is not None and "keep" in units:
        units = units[units["keep"] != 0]

    pieces, unaligned = [], 0
    for path, rows in trials.groupby(NWB_COLUMN, sort=False):
        moments = rows[align].to_numpy(dtype=float)
        unaligned += int(np.isnan(moments).sum())
        rows = rows[~np.isnan(moments)]
        if rows.empty:
            continue
        if units is None:
            wanted = None
        else:
            wanted = set(units.loc[units[NWB_COLUMN] == path, "unit_id"].astype(int))
        with NWBHDF5IO(str(path), "r") as io:
            stored = io.read().units
            if stored is None:
                continue
            log(f"  {Path(path).name}: {len(rows)} trial(s)")
            ids = stored.id[:]
            if "neuron_id" in stored.colnames:
                names = stored["neuron_id"].data[:]
            else:
                names = [""] * len(ids)
            for row, unit_id in enumerate(ids):
                if wanted is not None and int(unit_id) not in wanted:
                    continue
                times = np.asarray(stored.get_unit_spike_times(row))
                pieces += _windows(times, rows, align, before, after, int(unit_id),
                                   names[row] or np.nan)

    if unaligned:
        log(f"  {unaligned} trial(s) have no {align} and are left out")
    if pieces:
        spikes = pd.concat(pieces, ignore_index=True)
    else:
        spikes = pd.DataFrame(columns=["session_id", "trial", "unit_id", "neuron_id", "t",
                                       "t_rel"])
    # Scalars only: pandas compares attrs on every concat, and a table in here would
    # make that comparison ambiguous and raise.
    spikes.attrs.update(align=align, before=before, after=after)
    log(f"  {len(spikes):,} spike(s)")
    return spikes


def _windows(times, rows, align, before, after, unit_id, neuron_id):
    """
    One unit's spikes inside every trial's window, by binary search, all at once: a
    list holding one DataFrame, or empty when the unit fired in none of them.
    """
    moments = rows[align].to_numpy(dtype=float)
    low = np.searchsorted(times, moments - before, "left")
    high = np.searchsorted(times, moments + after, "right")
    counts = np.maximum(high - low, 0)
    if not counts.sum():
        return []
    which = np.repeat(np.arange(len(moments)), counts)
    picked = np.concatenate([np.arange(a, b) for a, b in zip(low, high) if b > a])
    here = times[picked]
    return [pd.DataFrame({"session_id": rows["session_id"].to_numpy()[which],
                          "trial": rows["trial"].to_numpy()[which],
                          "unit_id": unit_id, "neuron_id": neuron_id,
                          "t": here, "t_rel": here - moments[which]})]


def lump_neurons(spikes, window_s=COINCIDENT_S):
    """
    One train per neuron per session, where the matching named two units as one cell.

    The units' spikes are pooled; a spike within ``window_s`` of one already taken from
    another unit of the same neuron is dropped, since two units sharing a spike is the
    sorter catching it twice rather than the cell firing twice. Spikes are taken unit by
    unit in ``unit_id`` order, so which copy survives is fixed.

    Unnamed units pass through unchanged. ``unit_id`` stays on each row, saying where
    the spike came from. ``attrs["coincident_dropped"]`` counts what was dropped, per
    (session, neuron).
    """
    attrs = dict(spikes.attrs)
    spikes = spikes.copy()
    spikes.attrs = {}
    named = spikes.dropna(subset=["neuron_id"])
    shared = (named.groupby(["session_id", "neuron_id"])["unit_id"].nunique() > 1)
    shared = set(shared[shared].index)
    if not shared:
        spikes.attrs = {**attrs, "coincident_dropped": {}}
        return spikes

    is_shared = [(s, n) in shared for s, n in zip(spikes["session_id"], spikes["neuron_id"])]
    keep_rows = [spikes[~np.array(is_shared)]]
    # (session_id, neuron_id) -> spikes dropped as caught twice
    dropped = {}
    for (session_id, neuron_id, trial), group in spikes[np.array(is_shared)].groupby(
            ["session_id", "neuron_id", "trial"], sort=False):
        taken = np.empty(0)
        for _, unit in group.sort_values("unit_id").groupby("unit_id", sort=True):
            times = unit["t"].to_numpy()
            if len(taken):
                nearest = np.searchsorted(taken, times)
                left = np.abs(times - taken[np.clip(nearest - 1, 0, len(taken) - 1)])
                right = np.abs(times - taken[np.clip(nearest, 0, len(taken) - 1)])
                clash = np.minimum(left, right) <= window_s
            else:
                clash = np.zeros(len(times), dtype=bool)
            keep_rows.append(unit[~clash])
            key = (session_id, neuron_id)
            dropped[key] = dropped.get(key, 0) + int(clash.sum())
            taken = np.sort(np.concatenate((taken, times[~clash])))

    lumped = pd.concat(keep_rows).sort_values(["session_id", "trial", "t"]).reset_index(drop=True)
    lumped.attrs = {**attrs, "coincident_dropped": dropped}
    return lumped


def describe_units(units, columns=()):
    """
    Per session: how many units, how many named, and how many marked to keep. Each name
    in ``columns`` (any iterable of unit columns) adds that column's median.
    """
    medians = {column: (column, "median") for column in columns}
    summary = units.groupby("session_id").agg(
        units=("unit_id", "size"), named=("neuron_id", "count"),
        kept=("keep", lambda k: int((k != 0).sum())),
        spikes=("n_spikes", "sum"), **medians)
    return summary.reset_index()


# =====================================================================
# Every neuron's rate on every trial, binned through the trial
# =====================================================================

def neuron_key(units):
    """A neuron's name, or ``<session>:<unit>`` for a unit the matching did not name."""
    fallback = units["session_id"].astype(str) + ":" + units["unit_id"].astype(str)
    return units["neuron_id"].where(units["neuron_id"].notna(), fallback)


def bin_trials(trials, units=None, *, align="start_time", until=None, before=1.0,
               after=1.0, bin_s=0.1, warped_bins=10, baseline=(-1.0, 0.0),
               coincident_s=COINCIDENT_S, log_fn=None):
    """
    Every neuron's rate through every trial, in bins.

    Two ways to lay the trial out:

    * ``until=None`` — fixed time: ``before`` seconds ahead of ``align`` to ``after``
      seconds past it, in ``bin_s`` bins. ``x`` is seconds from ``align``.
    * ``until="t_response"`` — **stretched between two moments**: ``before`` seconds
      ahead of ``align`` in ``bin_s`` bins, then ``align`` → ``until`` cut into
      ``warped_bins`` equal parts whatever its length, then ``after`` seconds past
      ``until``. ``x`` is seconds before ``align`` (negative), the **fraction** of the
      way from ``align`` to ``until`` (0–1), then 1 + seconds after ``until``. Trials of
      different lengths then line up by phase. A trial with no ``until`` (a timeout has
      no response), or one where it comes before ``align``, is left out and counted.

    Args:
        trials: rows of a :func:`~.sessions.load_sessions` table — pass the ones you mean;
            QC and conditions are the caller's, as everywhere.
        units: rows of :func:`load_units`; default every unit. ``keep == 0`` is left out.
        baseline: a ``(start, stop)`` pair of seconds relative to ``start_time`` (the
            cue), whatever ``align`` is. Each row's rate there is ``baseline_hz``.
        coincident_s: two units of one session named as one neuron are pooled into one
            train, dropping a spike within this of one already taken — the rule
            :func:`lump_neurons` uses.

    A neuron-trial row counts only where the probe was recording through all of it —
    the binned span and the baseline; outside that a neuron is unobserved, not silent.

    Returns:
        A :class:`TrialRates`. ``layout`` records how it was binned; ``dropped`` what was
        left out and why.
    """
    log = log_fn or _say_nothing
    _needs(trials, NWB_COLUMN, "bin_trials")
    for column in (align, until, "start_time"):
        if column is not None and column not in trials:
            raise KeyError(f"no column {column!r} on the trials")
    units = load_units(trials) if units is None else units
    units = units[units["keep"] != 0] if "keep" in units else units
    units = units.assign(neuron=neuron_key(units))
    # reason -> how many were left out for it
    dropped = {}

    # ---- 1. The trials that have both moments ---------------------------
    moment = trials[align].to_numpy(float)
    usable = ~np.isnan(moment)
    dropped[f"no {align}"] = int((~usable).sum())
    if until is not None:
        end = trials[until].to_numpy(float)
        bad = usable & ~(end > moment)
        dropped[f"no {until} after {align}"] = int(bad.sum())
        usable &= ~bad
    trials = trials[usable]

    # ---- 2. Each trial's bins, and the span it needs recorded -----------
    edges, x, x_edges = _layout(trials, align, until, before, after, bin_s, warped_bins)
    cue = trials["start_time"].to_numpy(float)
    first = np.minimum(edges[:, 0], cue + baseline[0])
    last = np.maximum(edges[:, -1], cue + baseline[1])

    # ---- 3. Every neuron's counts in those bins, session by session -----
    rows, rates, outside = [], [], 0
    for path, here in units.groupby(NWB_COLUMN, sort=False):
        session_id = here["session_id"].iloc[0]
        which = np.flatnonzero(trials["session_id"].to_numpy() == session_id)
        if not len(which):
            continue
        inside = ((first[which] >= here["obs_start"].max())
                  & (last[which] <= here["obs_stop"].min()))
        outside += int((~inside).sum()) * here["neuron"].nunique()
        which = which[inside]
        if not len(which):
            continue
        log(f"  {Path(path).name}: {len(which)} trial(s), {here['neuron'].nunique()} neuron(s)")
        for neuron, train in _trains(path, here, coincident_s).items():
            counts = np.diff(np.searchsorted(train, edges[which]), axis=1)
            rates.append(counts / np.diff(edges[which], axis=1))
            in_baseline = (np.searchsorted(train, cue[which] + baseline[1])
                           - np.searchsorted(train, cue[which] + baseline[0]))
            rows.append(pd.DataFrame({
                "neuron": neuron, "session_id": session_id,
                "trial": trials["trial"].to_numpy()[which],
                "baseline_hz": in_baseline / (baseline[1] - baseline[0])}))
    dropped["neuron-trials outside the probe's recorded span"] = outside
    if not rows:
        raise ValueError("no neuron has a trial inside its recorded span")

    # ---- 4. The result --------------------------------------------------
    if until:
        layout_warped_bins = warped_bins
    else:
        layout_warped_bins = None
    layout = {"align": align, "until": until, "before": before, "after": after,
              "bin_s": bin_s, "warped_bins": layout_warped_bins,
              "baseline_s_from_cue": list(baseline), "coincident_s": coincident_s}
    reasons = {reason: n for reason, n in dropped.items() if n}
    out = TrialRates(rows=pd.concat(rows, ignore_index=True), rates=np.vstack(rates), x=x,
                     x_edges=x_edges, layout=layout, dropped=reasons)
    log(f"  {out.rows['neuron'].nunique()} neuron(s), {len(out.rows):,} neuron-trial(s)")
    return out


def _layout(trials, align, until, before, after, bin_s, warped_bins):
    """
    Each trial's bin edges on the DAQ clock, and where each bin's centre and edges sit
    on the axis: ``(edges, x, x_edges)``.
    """
    start = trials[align].to_numpy(float)[:, None]
    n_before = int(round(before / bin_s))
    n_after = int(round(after / bin_s))
    half = bin_s / 2
    if until is None:
        offsets = bin_s * np.arange(-n_before, n_after + 1)
        return start + offsets, offsets[:-1] + half, offsets
    end = trials[until].to_numpy(float)[:, None]
    lead = start + bin_s * np.arange(-n_before, 0)
    middle = start + (end - start) * np.arange(warped_bins) / warped_bins
    tail = end + bin_s * np.arange(n_after + 1)
    x_edges = np.concatenate([bin_s * np.arange(-n_before, 0),
                              np.arange(warped_bins) / warped_bins,
                              1 + bin_s * np.arange(n_after + 1)])
    x = (x_edges[:-1] + x_edges[1:]) / 2
    return np.hstack([lead, middle, tail]), x, x_edges


def _trains(path, units, coincident_s):
    """
    Each neuron's whole spike train in one session, as neuron -> spike times. Units named
    as one neuron are taken in ``unit_id`` order, each spike within ``coincident_s`` of
    one already taken dropped.
    """
    with NWBHDF5IO(str(path), "r") as io:
        stored = io.read().units
        position = {int(unit): row for row, unit in enumerate(stored.id[:])}
        times = {int(unit): np.asarray(stored.get_unit_spike_times(position[int(unit)]))
                 for unit in units["unit_id"]}
    out = {}
    for neuron, group in units.groupby("neuron"):
        taken = np.empty(0)
        for unit in sorted(group["unit_id"].astype(int)):
            train = times[unit]
            if len(taken) and len(train):
                nearest = np.searchsorted(taken, train)
                left = np.abs(train - taken[np.clip(nearest - 1, 0, len(taken) - 1)])
                right = np.abs(train - taken[np.clip(nearest, 0, len(taken) - 1)])
                train = train[np.minimum(left, right) > coincident_s]
            taken = np.sort(np.concatenate((taken, train)))
        out[neuron] = taken
    return out


def describe_neurons(units):
    """
    One row per neuron, from its units: where it was and how often it was seen.

    ``shank`` and ``depth_um`` are the neuron's own (its largest waveform's) where
    ``neurons.csv`` has them, otherwise its units'. These are the columns an ephys
    plotter can split neurons by, next to any you add — a ``cell_type`` worked out in a
    notebook is joined on ``neuron``.
    """
    units = units[units["keep"] != 0] if "keep" in units else units
    units = units.assign(neuron=neuron_key(units))

    # The neuron's own column where neurons.csv gave one, else the unit's.
    shank = "_shank" if "_shank" in units else "shank"
    depth = "_depth_um" if "_depth_um" in units else "depth_um"

    # The mouse from the index where it is there, else from the session ID.
    from_id = units["session_id"].astype(str).str.split("_").str[2]
    if "_mouse_id" in units:
        mouse = units["_mouse_id"]
    else:
        mouse = from_id
    units = units.assign(_mouse=mouse.fillna(from_id))

    out = units.groupby("neuron").agg(
        mouse_id=("_mouse", "first"), n_sessions=("session_id", "nunique"),
        n_units=("unit_id", "size"),
        shank=(shank, "first"),
        depth_um=(depth, "median"),
        firing_rate_hz=("firing_rate_hz", "median")).reset_index()
    if "_waveform_width_ms" in units:
        out = out.merge(units.groupby("neuron")["_waveform_width_ms"].first()
                        .rename("waveform_width_ms").reset_index(), on="neuron")
    return out


# =====================================================================
# What comes out of bin_trials
# =====================================================================

@dataclass
class TrialRates:
    """
    One row per neuron per trial: its rate in every bin, and in the baseline before the cue.

    ``rows`` says whose each row is (``neuron``, ``session_id``, ``trial``) and holds its
    ``baseline_hz``; ``rates`` holds the binned rates (Hz), one row of it per row of
    ``rows``; ``x`` is where each bin's centre sits on the axis and ``x_edges`` where its
    edges do. A trial on which a neuron did not
    fire is a row of zeros, not a missing row — a silent trial is data.
    """

    rows: pd.DataFrame
    rates: np.ndarray
    x: np.ndarray
    x_edges: np.ndarray
    layout: dict
    dropped: dict = field(default_factory=dict)

    @property
    def warped(self):
        """Whether the middle of the axis is a fraction of ``align`` → ``until``."""
        return self.layout.get("until") is not None

    def window(self, start, stop):
        """Each row's mean rate over the bins centred in ``[start, stop)``, in axis units."""
        inside = (self.x >= start) & (self.x < stop)
        if not inside.any():
            raise ValueError(f"no bins between {start} and {stop}; the axis runs "
                             f"{self.x[0]:.3g} to {self.x[-1]:.3g}")
        return self.rates[:, inside].mean(axis=1)

    def baseline(self, rows=None):
        """
        Each neuron's baseline: the mean and the standard deviation across trials of its
        rate in the baseline window. ``rows`` (a slice of ``self.rows``) restricts which
        trials count (default all).

        ``sd`` is across trials with ``ddof=1`` — a sample of trials, not a population of
        replicates, and the same spread ``population_activity`` has always divided by.
        """
        rows = self.rows if rows is None else rows
        return (rows.groupby("neuron")["baseline_hz"]
                .agg(baseline_mean="mean", baseline_sd="std", n_trials="size")
                .reset_index())
