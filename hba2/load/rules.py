"""
The one way of saying which rows you mean.
==========================================

Three places ask the same question — which sessions, which trials pass QC, which trials
are in a group — so they ask it in one vocabulary::

    include=True                     equal to
    mouse=["M1", "M2"]             any of
    not_mouse="test"                 anything but
    min_duration=1800                at or above
    max_abs_frame_lag_s=0.1          size at or below, sign ignored

=========================  ==================================================
``min_<column>``           keep rows at or above a value
``max_<column>``           keep rows at or below
``max_abs_<column>``       keep rows whose size is at or below, sign ignored — a lag of
                           -30 s is as bad as +30 s
``min_abs_<column>``       keep rows whose size is at or above — cues at least 30 degrees
                           off centre, either side
``<column>=value``         keep rows equal to it; a list means any of them, and ``None``
                           in that list means "or blank"
``not_<column>=value``     the other way round — drop those rows, keep the rest
=========================  ==================================================

**Blanks never pass.** A trial with no ``ear_likelihood`` fails ``min_ear_likelihood``:
there is no reading, so there is nothing to trust.

``describe_trials(trials)`` and ``describe_sessions(cohort)`` list what there is to
filter by, which is the menu for this.
"""

import pandas as pd

#: How a rule's name says what it does. Longest first, so ``max_abs_`` is not read as
#: ``max_`` of a column called ``abs_...``.
PREFIXES = ("max_abs_", "min_abs_", "min_", "max_")

#: Every prefix a rule name can carry, for stripping one off to name its column.
ALL_PREFIXES = PREFIXES + ("not_",)


def passes(frame, rule, value, column_for=None):
    """
    Which rows one rule keeps, as a boolean Series. Anything not comparable — a blank —
    does not pass.

    ``column_for(frame, name, rule)`` says which column a rule's name means, so the same
    rules work on the session index, where ``mouse`` means ``_mouse_id``, as on a table of
    trials, where names are exact. Defaults to exact.
    """
    column_for = column_for or _column
    for prefix in PREFIXES:
        if rule.startswith(prefix):
            column = column_for(frame, rule[len(prefix):], rule)
            values = frame[column]
            numbers = pd.to_numeric(values, errors="coerce")
            # Dates and names order perfectly well as text — "2026-06-01" sorts the way
            # you mean — so only fall back to numbers when the column holds them.
            comparable = numbers if numbers.notna().any() else values

            if prefix == "min_":
                return comparable >= value
            if prefix == "max_":
                return comparable <= value
            if comparable is not numbers:
                raise TypeError(f"{rule!r} needs a numeric column, and {column!r} holds "
                                f"{values.dtype} — there is no size to take.")
            if prefix == "min_abs_":
                return numbers.abs() >= value
            return numbers.abs() <= value

    if rule.startswith("not_"):
        return ~_equals(frame, column_for(frame, rule[len("not_"):], rule), value)

    return _equals(frame, column_for(frame, rule, rule), value)


def matching(frame, conditions, column_for=None):
    """Rows satisfying every rule in a dict. An empty dict means every row."""
    match = pd.Series(True, index=frame.index)
    for rule, value in conditions.items():
        match &= passes(frame, rule, value, column_for=column_for)
    return match


def bare(rule):
    """A rule with its prefix taken off, for naming the column it was about."""
    for prefix in ALL_PREFIXES:
        if rule.startswith(prefix):
            return rule[len(prefix):]
    return rule


def _equals(frame, column, value):
    """Rows equal to a value, or to any of a list. ``None`` in that list means blank."""
    values = list(value) if isinstance(value, (list, tuple, set)) else [value]
    wanted = [item for item in values if item is not None]
    # A Categorical only compares against its own categories, and isin on a value that is
    # not one of them is False rather than an error, which is what we want here anyway.
    kept = frame[column].isin(wanted)
    if None in values:
        kept |= frame[column].isna()
    return kept


def _column(frame, name, rule):
    """The column a rule names, or an error that says what there is."""
    if name in frame:
        return name
    raise KeyError(
        f"{rule!r} names no column {name!r}. describe_trials(trials) lists them with "
        f"what they mean. Numeric ones here: "
        f"{', '.join(sorted(frame.select_dtypes('number').columns))}")
