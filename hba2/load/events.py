"""
Finding the moment inside a trial that a figure is actually about.
==================================================================

The trials table says when each trial began and ended. It does not say when the cue came
on, because **which channel counts as a cue is protocol-specific** — that is the rule the
events table is built on, where a transition is named for the hardware that made it
(``led_on``, ``sensor_touch``) and never for what it meant.

So the meaning is asserted here, once, where you can see it::

    events = load_events(trials, before=1.0)          # the raw transitions

    trials = attach_events(trials,
        cue      = first("led_on",       within=(0.5, 0)),
        touch    = first("sensor_touch", after="cue"),
        reward   = first("valve_open",   after="touch"),
        touches  = count("sensor_touch"),
    )

which adds ``t_cue`` and ``port_cue``, ``t_touch`` and ``port_touch``, and so on. Not
``response``: the table already has ``t_response``, the chosen port's beam break, and
a selector may not replace a column it did not make.
The port comes free, because it is already the event's own column — so "which port did
it touch first" needs nothing extra.

**``after=`` is the one that matters.** A response is the first sensor touch *after the
cue*, not the first in the trial: an animal already resting on a sensor when the marker
fires would otherwise give a response time of zero, and that trial would look like the
fastest in the session.

**Windows reach outside the trial.** ``within=(0.5, 0)`` searches from half a second
before the marker to the trial end, which is what a cue that leads the marker needs.
Both numbers are positive extents, the same way :func:`load_frames` reads them.

Nothing found gives NaN, never zero.

This is a separate call from :func:`load_sessions` for the same reason per-frame data is:
a session has about thirty times more events than trials, and most figures need none of
them. It reads each file once and takes the trials you hand it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO

from .sessions import NWB_COLUMN, _needs, _say_nothing

#: Where the hardware transitions live in the file.
TABLE = ("behaviour", "events")

#: How a picked event is written onto the trials table.
TIME = "t_{name}"
PORT = "port_{name}"
COUNT = "n_{name}"


@dataclass(frozen=True)
class Selector:
    """One event to find per trial, and where to look for it."""

    event: str
    pick: str = "first"
    n: int = 0
    within: tuple[float, float] = (0.0, 0.0)
    after: str | None = None
    channel: str | None = None
    port: int | None = None

    def how(self):
        """One line for the provenance record."""
        parts = [f"{self.pick} {self.event!r}"]
        if self.channel:
            parts.append(f"on {self.channel}")
        if self.port is not None:
            parts.append(f"at port {self.port}")
        if self.after:
            parts.append(f"after {self.after}")
        before, after = self.within
        if before or after:
            parts.append(f"within -{before:g}s to +{after:g}s of the trial")
        return " ".join(parts)


def first(event, **where):
    """The earliest matching event in the window."""
    return Selector(event=event, pick="first", **where)


def last(event, **where):
    """The latest matching event in the window."""
    return Selector(event=event, pick="last", **where)


def nth(event, n, **where):
    """The ``n``-th matching event, 0-based. Negative counts from the end."""
    return Selector(event=event, pick="nth", n=n, **where)


def count(event, **where):
    """How many matched — for revisiting, perseveration, how busy a trial was."""
    return Selector(event=event, pick="count", **where)


def _read(path):
    """One session's events table, in the order it was written — which is by time."""
    with NWBHDF5IO(str(path), "r") as io:
        module, table = TABLE
        return io.read().processing[module][table].to_dataframe().reset_index(drop=True)


def load_events(trials, *, before=0.0, after=0.0, events=None, log_fn=None):
    """
    Every hardware transition belonging to the trials you hand it.

    Args:
        trials: rows of a :func:`load_sessions` table.
        before: seconds before each trial's start to include.
        after: seconds after each trial's end.
        events: only these event names (a tuple), for when a trial's LEDs are all you
            want.
        log_fn: called with a line per file opened.

    Returns:
        One row per event per trial: ``session_id``, ``trial``, ``t``, ``t_from_start``,
        ``t_from_end``, ``event``, ``channel``, ``port``, ``in_trial``,
        ``trial_containing``.

    ``trial`` is the trial whose window this row was found for; ``trial_containing`` is
    the trial the event actually fell inside, as post-processing recorded it, and
    ``in_trial`` is 0 where it fell between trials. They differ exactly when
    ``before``/``after`` reach outside the marker — which is the case a cue leading its
    trial needs, so the distinction is kept rather than collapsed.

    An event inside the widened window of two trials appears once for each, because it
    is genuinely relevant to both.
    """
    log = log_fn or _say_nothing
    _needs(trials, NWB_COLUMN, "load_events")

    pieces = []
    for path, rows in trials.groupby(NWB_COLUMN, sort=False):
        table = _read(path)
        log(f"  {path.rsplit('/', 1)[-1]}: {len(table):,} events, {len(rows)} trial(s)")
        if events is not None:
            table = table[table["event"].isin(events)]
        times = table["t"].to_numpy(dtype=float)

        for _, trial in rows.iterrows():
            start = float(trial["start_time"]) - before
            stop = float(trial["stop_time"]) + after
            low = int(np.searchsorted(times, start, "left"))
            high = int(np.searchsorted(times, stop, "right"))
            if high <= low:
                continue
            piece = table.iloc[low:high].copy()
            # The file's own column says which trial contains the event; the one this
            # row is attached to is the trial whose window was searched, and a widened
            # window makes those different things.
            piece = piece.rename(columns={"trial": "trial_containing"})
            piece.insert(0, "trial", trial["trial"])
            piece.insert(0, "session_id", trial["session_id"])
            piece["t_from_start"] = piece["t"] - float(trial["start_time"])
            piece["t_from_end"] = piece["t"] - float(trial["stop_time"])
            pieces.append(piece)

    if not pieces:
        return pd.DataFrame(columns=["session_id", "trial", "t", "t_from_start",
                                     "t_from_end", "event", "channel", "port",
                                     "in_trial", "trial_containing"])
    found = pd.concat(pieces, ignore_index=True)
    log(f"  {len(found):,} events across {trials['session_id'].nunique()} session(s)")
    return found


def attach_events(trials, *, log_fn=None, **selectors):
    """
    Pick one event per trial per selector, and write it onto the table. Each selector
    is a :class:`Selector`, made by :func:`first`, :func:`last`, :func:`nth` or
    :func:`count`.

    Selectors are applied in the order given, so ``after=`` may name any earlier one.
    Each adds ``t_<name>`` and ``port_<name>``, or ``n_<name>`` for a
    :func:`count`, and records what it meant in ``attrs["tags"]`` so the provenance
    beside a figure says how its cue was defined.

    A selector may replace only columns an earlier ``attach_events`` made, so re-running
    a notebook cell is fine. Any other column of that name is the file's own —
    ``t_response`` comes with every trial — and replacing it silently would change what
    a figure measured, so that raises instead.
    """
    if not selectors:
        raise ValueError("attach_events needs at least one selector, e.g. "
                         "cue=first('led_on', within=(0.5, 0))")
    for name, selector in selectors.items():
        if not isinstance(selector, Selector):
            raise TypeError(
                f"{name}= is {type(selector).__name__}, not a selector. Use first(...), "
                "last(...), nth(...) or count(...).")

    made_here = trials.attrs.get("tags", {})
    for name, selector in selectors.items():
        if selector.pick == "count":
            columns = [COUNT.format(name=name)]
        else:
            columns = [TIME.format(name=name), PORT.format(name=name)]
        taken = [c for c in columns if c in trials]
        if taken and not str(made_here.get(name, "")).startswith("event:"):
            raise ValueError(
                f"{name}= would replace {', '.join(taken)}, which the table already has "
                f"and attach_events did not make. Give the selector another name.")

    # One read, wide enough for every selector, rather than one per selector.
    reach_before = max(selector.within[0] for selector in selectors.values())
    reach_after = max(selector.within[1] for selector in selectors.values())
    events = load_events(trials, before=reach_before, after=reach_after,
                         log_fn=log_fn)

    attached = trials.copy()
    tags = dict(trials.attrs.get("tags", {}))
    for name, selector in selectors.items():
        attached = _apply(attached, events, name, selector)
        tags[name] = f"event: {selector.how()}"
    attached.attrs["tags"] = tags
    return attached


def describe_events(trials):
    """
    What kinds of event this data holds, and on which channels — the menu for a selector.

    Read from the files rather than from documentation, because which channels were
    wired up is a fact about the recording.
    """
    _needs(trials, NWB_COLUMN, "describe_events")
    table = _read(trials[NWB_COLUMN].iloc[0])
    summary = (table.groupby(["event", "channel"], sort=True)
               .agg(n=("t", "size"), ports=("port", lambda values:
                                            ", ".join(map(str, sorted(set(values))))))
               .reset_index())
    return summary


# =====================================================================
# The steps
# =====================================================================

def _apply(trials, events, name, selector):
    """
    One selector's columns, joined back onto the trials by identity. Columns of the
    same names already there are dropped first, so the merge cannot leave both as
    ``_x``/``_y`` and neither under the name asked for.
    """
    matched = _matching(trials, events, name, selector)
    keys = ["session_id", "trial"]
    if selector.pick == "count":
        replacing = [COUNT.format(name=name)]
    else:
        replacing = [TIME.format(name=name), PORT.format(name=name)]
    trials = trials.drop(columns=[c for c in replacing if c in trials])

    if selector.pick == "count":
        found = matched.groupby(keys, sort=False).size().rename(COUNT.format(name=name))
        joined = trials.merge(found.reset_index(), on=keys, how="left")
        joined[COUNT.format(name=name)] = joined[COUNT.format(name=name)].fillna(0).astype(int)
        return joined

    ordered = matched.sort_values(keys + ["t"], kind="stable")
    grouped = ordered.groupby(keys, sort=False)
    if selector.pick == "first":
        chosen = grouped.head(1)
    elif selector.pick == "last":
        chosen = grouped.tail(1)
    elif selector.pick == "nth":
        chosen = grouped.nth(selector.n)
    else:
        raise ValueError(f"pick must be first, last, nth or count, not {selector.pick!r}")

    chosen = chosen[keys + ["t", "port"]].rename(columns={
        "t": TIME.format(name=name), "port": PORT.format(name=name)})
    return trials.merge(chosen, on=keys, how="left")


def _matching(trials, events, name, selector):
    """The events one selector is allowed to choose from."""
    if events.empty:
        return events

    keep = events["event"] == selector.event
    if selector.channel is not None:
        keep &= events["channel"] == selector.channel
    if selector.port is not None:
        keep &= events["port"] == selector.port

    before, after = selector.within
    keep &= events["t_from_start"] >= -before
    keep &= events["t_from_end"] <= after
    matched = events[keep]

    if selector.after is not None:
        column = TIME.format(name=selector.after)
        if column not in trials:
            raise KeyError(
                f"{name}= is after {selector.after!r}, which has not been attached. "
                "Selectors run in the order given, so put it first.")
        earlier = trials[["session_id", "trial", column]]
        matched = matched.merge(earlier, on=["session_id", "trial"], how="left")
        # A trial whose earlier event was never found has nothing to be after, so it
        # gets nothing here either — rather than silently falling back to the window.
        matched = matched[matched["t"] > matched[column]]
        matched = matched.drop(columns=column)

    return matched
