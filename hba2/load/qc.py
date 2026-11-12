"""
Marking the trials that smell bad.
==================================

One sweep, once, at the top of a notebook, with the thresholds written where you can see
them::

    trials = qc_trials(
        all_trials,
        min_ear_likelihood=0.6,          # DeepLabCut was sure about the ears
        max_abs_frame_lag_s=0.1,         # a video frame near the moment asked about
        max_response_distance_px=100,    # the head was at the port it is said to touch
    )

    qc_trials: 10,935 trials → 9,459 keep, 1,476 marked
      min_ear_likelihood=0.6            366
      max_abs_frame_lag_s=0.1           108
      max_response_distance_px=100    1,002

**Nothing is removed.** The table that comes back is the same height as the one that went
in, with a ``keep`` column holding 1 or 0. Every plotter drops ``keep == 0`` before it
does anything else and never sees those trials again — so the effect is the same as
throwing them away, but three things become possible that deletion makes impossible:

*A trial that is junk as a **subject** is still real as **context**.* Anything that reads
a neighbouring trial — was the last one an error, how long was the gap — needs the
neighbour to still be there, even when the neighbour's pose was unusable. Under deletion,
"the previous trial" quietly becomes "the previous *surviving* trial", which is a
different and wrong number.

*QC bias becomes visible.* If pose quality fails more often on laser trials than on
control trials, the comparison is broken. As a column that is one ``groupby``; as a
separate table it is invisible, because nobody joins it back.

*What was excluded travels with the data*, into every table derived from it and into the
figures' saved numbers, rather than living in a second table that has to be carried
around beside the first.

``keep`` is **one gate, not several**. Which trials are worth using is a judgement about
the data, made once, in one place — a plot that wants a different judgement should say so
by re-running this, not by reaching past it.

``qc_failed`` names the first rule a trial fell to, for looking at what was marked. It is
a label, not a second gate.

**The trajectory is the other check on the pose**, and often the better one: a frame can
be low-confidence and exactly right. ``head_speed_px_s`` and ``turn_rate_deg_s`` say how
fast the head moved and turned around the trial start frame (``response_...`` at the
touch), so a label that jumped or a bearing that swapped ends is a limit on either::

    qc_trials(trials, max_head_speed_px_s=2500, max_turn_rate_deg_s=2700,
              max_abs_frame_lag_s=0.1)

They judge the moment, not the trial — what a mouse did up the wall mid-trial does not
spoil its start. For a time course, ``load_frames`` carries the same two per frame.

The numbers above are the previous library's: 0.6 for the ears, and 0.1 s — three frames —
for how far a looked-up frame may sit from its event. They are written at the call rather
than hidden in here, because they are a judgement about your data, not a fact about it.
"""

import pandas as pd

from .rules import passes

#: The one gate. 1 keeps a trial, 0 marks it. Plotters drop the zeros first thing.
KEEP = "keep"

#: The rule a marked trial fell to, empty where it passed. A label, not a gate.
REASON = "qc_failed"

#: Written onto the table, so anything later can say how it was checked.
RECORD = "qc"


def _say_nothing(message):
    """What qc_trials reports to when it has been silenced."""


def qc_trials(trials, *, log_fn=print, **rules):
    """
    Mark the trials nothing should be computed from.

    Args:
        trials: a :func:`load_sessions` table.
        log_fn: where the report goes. ``print`` by default, because a check you cannot
            see is a check you forget; ``None`` silences it.
        **rules: ``min_<column>``, ``max_<column>``, ``max_abs_<column>``, or
            ``<column>=value`` — see :mod:`.rules`. Applied in the order given, and the
            first one a trial fails is the one recorded against it.

    Returns:
        The same table, same height, with ``keep`` (1 or 0) and ``qc_failed`` added.
    """
    log = log_fn or _say_nothing
    checked = trials.copy()

    keep = pd.Series(True, index=checked.index)
    reason = pd.Series("", index=checked.index, dtype=object)
    report = {}

    for rule, value in rules.items():
        named = f"{rule}={value!r}"
        failed = ~passes(checked, rule, value)
        # Counted against every rule it fails, so the report adds up to more than the
        # number marked when a trial is bad in two ways — which is the honest reading of
        # "how many trials does this threshold cost me".
        report[named] = int(failed.sum())
        # Named for the first rule it fell to, so the label is stable when rules are
        # reordered around it.
        reason = reason.mask(failed & (reason == ""), named)
        keep &= ~failed

    checked[KEEP] = keep.astype(int)
    checked[REASON] = reason
    checked.attrs[RECORD] = report

    # The lines printed: the totals, then what each rule cost.
    total, keeping = len(checked), int(keep.sum())
    lines = [f"qc_trials: {total:,} trials → {keeping:,} keep, {total - keeping:,} marked"]
    width = max((len(rule) for rule in report), default=0)
    lines += [f"  {rule:<{width}}  {count:>7,}" for rule, count in report.items()]
    log("\n".join(lines))
    return checked


def kept(trials, who="this plot"):
    """
    The trials worth using — what every plotter calls before anything else.

    A table that has never been through :func:`qc_trials` passes through untouched rather
    than failing: QC is a judgement you may not have wanted to make yet, and a plot of raw
    trials is a legitimate thing to draw.
    """
    if KEEP not in trials:
        return trials
    good = trials[trials[KEEP] == 1]
    if good.empty:
        raise ValueError(
            f"{who} has no trials left — every one of the {len(trials):,} given is marked "
            f"keep == 0. Check the thresholds in qc_trials.")
    return good


def record(trials):
    """
    What QC did to this table, for the provenance beside a figure.

    Read from ``attrs`` where it survived, and counted off the ``qc_failed`` column where
    it did not — ``attrs`` is lost through a ``merge``, and a figure whose saved JSON
    quietly says the thresholds were ``null`` is worse than one that says nothing. The
    columns are the durable record; the attrs are only a convenience.
    """
    if trials.attrs.get(RECORD):
        return trials.attrs[RECORD]
    if REASON not in trials:
        return {}
    marked = trials.loc[trials[REASON].astype(str) != "", REASON]
    return {rule: int(count) for rule, count in marked.value_counts().items()}
