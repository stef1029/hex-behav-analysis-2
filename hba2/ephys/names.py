"""
Neuron names: which units, across sessions, are the same cell.

A matching run (UnitMatch, or anything that produces the same table) decides that unit
*a* of one session and unit *b* of another are one neuron. This turns its decisions into
a name per unit — ``E4_n0012`` — written beside each session's curation as
``neuron_names.csv``. Plotting then groups by ``neuron_id`` like any other column, and a
neuron seen in five sessions is five sessions' spikes under one name.

The input is a table with a row per (session, unit) the run considered::

    epoch, track, session, cluster_id[, n_sessions_in_track]

``track`` is the run's own label for a neuron, unique within an ``epoch`` — a stretch of
sessions with no probe movement, outside which nothing is matched. That is what
``unit_tracking/results/tracks_all.csv`` holds.

**Names are given across the whole table**, ordered by epoch then track, so writing names
for one session gives it the same names it would get in a run over all of them. A unit
that appears in no other session still gets a name: grouping by name then needs no
special case. Two units of one session may share a name — the run decided they are one
cell split in two — and the plotting layer handles that (``lump_neurons``).

**Every good unit gets a name**, including ones the run never saw (a session left out of
the matching, a unit promoted to good afterwards). They are named after the run's own
neurons, numbered on per mouse in session-then-unit order, with ``level`` set to
``unmatched`` so they cannot be mistaken for a match. That is what lets every neuron
carry labels in ``neurons.csv``.

``neuron_names.csv``, one row per unit the run named::

    unit_id, neuron_id, epoch, track, n_sessions, matching, level

``matching`` and ``level`` say which run and which of its settings made the name, so a
file read later says where its names came from.
"""

import re
from pathlib import Path

import pandas as pd

from ..cohort import CURATION, Cohort

#: ``neuron_names.csv``'s columns, in order — see the module docstring.
COLUMNS = ("unit_id", "neuron_id", "epoch", "track", "n_sessions", "matching", "level")


def _track_number(track):
    """``e5_t28`` -> 28, so track 9 sorts before track 10. -1 if it ends in no number."""
    found = re.search(r"(\d+)$", track)
    return int(found.group(1)) if found else -1


def name_tracks(tracks):
    """
    A ``neuron_id`` for every row of a matching run's table (a DataFrame).

    One name per (epoch, track), numbered per mouse in epoch-then-track order:
    ``<mouse>_n<4 digits>``. The mouse is the session ID's last part.
    """
    table = tracks.copy()
    table["mouse"] = table["session"].astype(str).str.split("_", n=2).str[2]
    table["_track_no"] = table["track"].astype(str).map(_track_number)
    keys = (table[["mouse", "epoch", "_track_no", "track"]].drop_duplicates()
            .sort_values(["mouse", "epoch", "_track_no", "track"]))
    keys["neuron_id"] = keys["mouse"] + "_n" + (
        keys.groupby("mouse").cumcount() + 1).map("{:04d}".format)
    named = table.merge(keys[["mouse", "epoch", "track", "neuron_id"]],
                        on=["mouse", "epoch", "track"], how="left")
    if "n_sessions_in_track" not in named:
        named["n_sessions_in_track"] = named.groupby("neuron_id")["session"].transform("nunique")
    return named.drop(columns="_track_no")


def _good(cluster_group):
    """The cluster IDs a ``cluster_group.tsv`` (a Path) labels good, as a set of ints."""
    labels = pd.read_csv(cluster_group, sep="\t")
    return set(labels.loc[labels["group"].astype(str).str.strip() == "good",
                          "cluster_id"].astype(int))


def write_names(cohort, tracks, *, sort, curation, matching, level):
    """
    Write ``neuron_names.csv`` into every session's curation.

    Args:
        cohort: where the sessions are — a :class:`Cohort`, or a Path or string.
        tracks: the matching run's table, as a DataFrame or a path (Path or string) to
            its CSV.
        sort, curation: whose units these are. The run must have matched units of this
            curation — a cluster ID means nothing in another one.
        matching: a name for the run, recorded in every row (``unitmatch_2026-09-07``).
        level: which of the run's settings made the tracks (``intermediate``, or
            ``intermediate+manual`` where tracks were regrouped by hand).

    Every session in the cohort with this curation gets a file, whether or not the run
    saw it: good units the run did not name get a single-session name (see the module
    docstring). Returns a dict of counts, and the mismatches worth reading: units the
    run named that the curation does not call good, which get no name.
    """
    if not isinstance(cohort, Cohort):
        cohort = Cohort(cohort)
    if isinstance(tracks, (str, Path)):
        table = pd.read_csv(tracks)
    else:
        table = tracks
    named = name_tracks(table)

    report = {"sessions_written": [], "sessions_not_in_cohort": [],
              "named_but_not_good": {}, "named_unmatched": {}}
    report["sessions_not_in_cohort"] = sorted(
        s for s in named["session"].unique() if s not in cohort)

    # ---- 1. Every session holding this curation --------------------------
    # In session order, which is what fixes the numbering of the unmatched names.
    holding = {}
    for record in cohort.with_files():
        if curation in record.sorts.get(sort, {}).get("curations", {}):
            holding[record.session_id] = record

    # ---- 2. Each one's names ---------------------------------------------
    # The run's names for its good units, then a single-session name each for the good
    # units it never matched, numbered on per mouse after the run's highest.
    #: mouse -> the highest name number used so far
    numbered = (named["neuron_id"].str.rsplit("_n", n=1).str[1].astype(int)
                .groupby(named["mouse"]).max().to_dict())
    #: session ID -> its neuron_names.csv rows
    rows_by_session = {}
    for session_id, record in holding.items():
        files = record.curation(sort, curation)
        good = _good(files["cluster_group"])
        rows = named[named["session"] == session_id]
        named_here = set(rows["cluster_id"].astype(int))
        if named_here - good:
            report["named_but_not_good"][session_id] = sorted(named_here - good)
            rows = rows[rows["cluster_id"].astype(int).isin(good)]

        out = pd.DataFrame({
            "unit_id": rows["cluster_id"].astype(int),
            "neuron_id": rows["neuron_id"],
            "epoch": rows["epoch"],
            "track": rows["track"],
            "n_sessions": rows["n_sessions_in_track"].astype(int),
            "matching": matching,
            "level": level,
        })
        unmatched = sorted(good - named_here)
        if unmatched:
            mouse = record.mouse_id
            first = numbered.get(mouse, 0)
            numbered[mouse] = first + len(unmatched)
            out = pd.concat([out, pd.DataFrame({
                "unit_id": unmatched,
                "neuron_id": [f"{mouse}_n{first + i + 1:04d}" for i in range(len(unmatched))],
                "epoch": "", "track": "", "n_sessions": 1,
                "matching": matching, "level": "unmatched",
            })], ignore_index=True)
            report["named_unmatched"][session_id] = unmatched
        rows_by_session[session_id] = out.sort_values("unit_id")

    # ---- 3. Write them ---------------------------------------------------
    # Only once every session's names are worked out, so a failure part way leaves
    # no session half-renamed against the others.
    for session_id, out in rows_by_session.items():
        folder = Path(holding[session_id].curation(sort, curation)["record"]).parent
        out.to_csv(folder / CURATION["names"], index=False, columns=list(COLUMNS))
        report["sessions_written"].append(session_id)
    return report


def name_unnamed(cohort, *, sort, curation, matching="none"):
    """
    Single-session names for exported sessions that have no ``neuron_names.csv`` yet.

    For a run with no matching table: sessions already named keep their names, and the
    rest get theirs numbered on per mouse after the highest name already in use, with
    ``level`` ``unmatched``. A later :func:`write_names` with a real matching run
    renames them; ``neurons.csv`` carries labels across by membership, so nothing typed
    against these names is lost.

    ``cohort`` is a :class:`Cohort`, or a Path or string. Returns a dict: the sessions
    written and how many units were named.
    """
    if not isinstance(cohort, Cohort):
        cohort = Cohort(cohort)

    # ---- 1. The highest name per mouse, and who is waiting for names -----
    #: mouse -> the highest name number already in use
    highest = {}
    waiting = []
    for record in cohort.with_files():
        files = record.sorts.get(sort, {}).get("curations", {}).get(curation)
        if not files or not files.get("units"):
            continue
        if files.get("names"):
            names = pd.read_csv(files["names"], dtype={"neuron_id": str})["neuron_id"]
            for name in names:
                mouse, _, number = name.rpartition("_n")
                if number.isdigit():
                    highest[mouse] = max(highest.get(mouse, 0), int(number))
        else:
            waiting.append((record, files))

    # ---- 2. Name them, numbering on --------------------------------------
    report = {"sessions_written": [], "units_named": 0}
    for record, files in waiting:
        good = sorted(_good(files["cluster_group"]))
        mouse = record.mouse_id
        first = highest.get(mouse, 0)
        highest[mouse] = first + len(good)
        pd.DataFrame({
            "unit_id": good,
            "neuron_id": [f"{mouse}_n{first + i + 1:04d}" for i in range(len(good))],
            "epoch": "", "track": "", "n_sessions": 1,
            "matching": matching, "level": "unmatched",
        }).to_csv(Path(files["record"]).parent / CURATION["names"], index=False,
                  columns=list(COLUMNS))
        report["sessions_written"].append(record.session_id)
        report["units_named"] += len(good)
    return report
