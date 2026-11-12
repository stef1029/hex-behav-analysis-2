"""
Scales readings placed on the DAQ clock without a pulse index — a one-off fix.
=============================================================================

**Run after post-processing**, on sessions whose ``scales.csv`` is untimed::

    python repair_scales_timing.py <cohort> [--dry-run [--report-dir DIR]] [--only ...]

What went wrong
---------------
Post-processing times each weight reading by the ``SCALES`` pulse it produced. Some rigs
never wired the scales into the DAQ, and their scales CSV has no pulse index, so every
reading is written untimed. The readings carry only the scales server's own clock
(``timestamp_s``), which starts at a different moment from the DAQ's.

What this does
--------------
It uses the rule the mice were trained on: **a trial starts once the mouse has stood on
the platform for one second.** So every "weight above threshold for at least a second"
(a *platform event*, timed at onset + 1 s) should sit just before a trial start on the
DAQ clock, by one constant offset — both clocks are the same PC's, so there is no drift
to fit, and fitting a slope only lets bad sessions look good.

1. Platform events at each threshold from 4 to 25 g; the threshold kept is the one whose
   events pair with the most trial starts.
2. Pairing is :func:`legacy.trials._pair` — the offset most pairs agree on, then an
   order-preserving alignment that lets either side skip — and the offset is the median
   of the paired differences, trimmed to those within ``TRIM_S`` of it, three times.
3. The rig PC's clock can step mid-session (seen: 1.35 s). Trials the first offset does
   not explain are paired again on their own; a second offset with at least
   ``MIN_PAIRS`` pairs becomes a second stretch of the session, split where the paired
   trials change from one to the other.
4. **Applied only when all of these hold**, otherwise ``scales.csv`` is left untimed and
   the reason recorded: at least ``MIN_PAIRS`` pairs and ``MIN_FRACTION`` of trials
   paired; the median absolute residual at most ``MAX_MAD_S``; the offsets fitted on the
   first and second half of the pairs agree within ``MAX_HALVES_S``.

The offset should be the same for every session recorded the same way, so the cohort
step also checks each against the others: one far from the cohort's usual value is
accepted only if another session from the same launch agrees with it.

What it writes
--------------
``scales.csv`` with ``t`` filled, through post-processing's own writer; the untimed one
kept as ``scales_unplaced.csv``; ``session.json`` → ``scales_timing`` with the offset,
threshold, pairs and every check; and a picture per session in the cohort's
``alignment_report/scales/``. Times placed this way are the fit's, not a pulse's: good to
the tens of milliseconds the checks report, not to the DAQ's microseconds.
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from hba2 import Cohort  # noqa: E402
from hba2.legacy import trials as legacy_trials  # noqa: E402
from hba2.postprocessing import outputs  # noqa: E402
from hba2.postprocessing.sources import scales as scales_source  # noqa: E402
from hba2.postprocessing.utils.clock import Placement  # noqa: E402

HOLD_S = 1.0
THRESHOLDS_G = np.arange(4.0, 26.0, 1.0)
TRIM_S = 0.15
MIN_PAIRS = 20
MIN_FRACTION = 0.4
MAX_MAD_S = 0.030
MAX_HALVES_S = 0.030
#: A later stretch is real only as a block: in its own span it pairs at least this share
#: of trials, and the stretches before it pair at most ``BLOCK_OTHERS`` there. Platform
#: events are dense enough that 20 chance pairs scattered over a session are easy to find.
BLOCK_OWN = 0.6
BLOCK_OTHERS = 0.2
#: An offset further than this from the cohort's median needs a launch-mate to agree.
COHORT_BAND_S = 2.5
LAUNCH_AGREE_S = 0.5

BY = "hba2/postprocessing/repairs/repair_scales_timing.py"
UNPLACED = "scales_unplaced.csv"


def _launch(session_id):
    """The launch a session belongs to: its ID less the mouse."""
    return session_id.rsplit("_", 1)[0]


def repair_scales_timing(cohort, *, only=None, dry_run=False, report_dir=None,
                         workers=16, log=print):
    """
    Fit every processed session with a scales CSV, check each against the cohort, and
    place the ones that pass. Returns one row per session.

    ``cohort`` is a :class:`Cohort`, or its folder as a Path or a string; ``only`` a list
    of session IDs to limit it to. The report (a picture per session, the
    ``scales_timing.csv`` table and a summary picture) goes to ``report_dir``, or by
    default to the cohort's ``alignment_report/scales/``. A dry run writes nothing into
    the cohort: no session's files, and its report only when ``report_dir`` is given.
    """
    # ---- 1. Fit every session -----------------------------------------------
    cohort = cohort if isinstance(cohort, Cohort) else Cohort(cohort)
    records = [r for r in cohort if r.is_processed and r.raw.get("scales_csv") is not None
               and (only is None or r.session_id in only)]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(workers) as pool:
        results = list(pool.map(fit_session, records))

    # ---- 2. The cohort check: an offset far from the usual needs a launch-mate to agree
    passed = [r for r in results if r.get("passed")]
    usual = float(np.median([r["offset_s"] for r in passed])) if passed else np.nan
    # A launch-mate counts if its own fit is clean, even when it had too few trials to pass.
    clean = [r for r in results if "offset_s" in r and r["residual_mad_ms"] <= MAX_MAD_S * 1000]
    for r in passed:
        if abs(r["offset_s"] - usual) > COHORT_BAND_S:
            mates = [m for m in clean if m is not r and _launch(m["session"]) == _launch(r["session"])
                     and abs(m["offset_s"] - r["offset_s"]) <= LAUNCH_AGREE_S]
            if not mates:
                r["passed"] = False
                r["checks"]["cohort"] = False
                r["reason"] = (f"offset {r['offset_s']:.2f} s is far from the cohort's "
                               f"{usual:.2f} s and no session from the same launch agrees")

    # ---- 3. Apply the ones that passed, and draw each ---------------------------
    # Where the report goes: a dry run's only where asked, so it touches nothing.
    if report_dir is not None:
        report = Path(report_dir)
    elif dry_run:
        report = None
    else:
        report = cohort.base / "alignment_report" / "scales"
    by_id = {r.session_id: r for r in records}
    for r in results:
        if r.get("passed"):
            r["applied"] = not dry_run
            if not dry_run:
                apply(by_id[r["session"]], r)
        elif "checks" in r and "reason" not in r:
            r["reason"] = "failed: " + ", ".join(k for k, ok in r["checks"].items() if not ok)
        if "_plot" in r and report is not None:
            draw(r, report / f"{r['session']}.png")

    # ---- 4. The cohort's table and summary picture ------------------------------
    rows = []
    for r in results:
        rows.append({k: v for k, v in r.items() if not k.startswith("_")
                     and k not in ("checks", "stretches")})
    table = pd.DataFrame(rows)
    table["n_stretches"] = [len(r.get("stretches", [])) for r in results]
    if report is not None:
        report.mkdir(parents=True, exist_ok=True)
        table.to_csv(report / "scales_timing.csv", index=False)
        _summary_plot(table, usual, report / "_summary.png")
    log(f"{cohort.base.name}: scales placed in {int(table.applied.sum())} of {len(table)} "
        f"session(s); cohort offset {usual:.3f} s")
    return table


def fit_session(record):
    """The fit and its checks, without writing anything."""
    # ---- 1. Read ------------------------------------------------------------
    scales = scales_source.read(record.path("scales_csv"))
    trials = pd.read_csv(record.path("trials"))
    starts = np.sort(trials["t_start"].dropna().to_numpy())
    t, w = scales.timestamps, scales.weights
    valid = t > 0
    result = {"session": record.session_id, "n_trials": int(len(starts)),
              "n_readings": int(len(t)), "applied": False}
    if len(starts) < MIN_PAIRS or valid.sum() < 100:
        result["reason"] = "too few trials or readings"
        return result

    # ---- 2. The threshold whose platform events pair with the most trials --------
    best = None
    for threshold in THRESHOLDS_G:
        events = platform_events(t[valid], w[valid], threshold)
        if len(events) < MIN_PAIRS:
            continue
        stretches = fit_offsets(starts, events)
        n = sum(len(s["trials"]) for s in stretches)
        if stretches and (best is None or n > best[0]):
            best = (n, threshold, events, stretches)
    if best is None:
        result["reason"] = "no threshold gives enough platform events paired with trials"
        return result

    # ---- 3. The main stretch, and whether its offset holds over both halves -------
    n, threshold, events, stretches = best
    main = max(stretches, key=lambda s: len(s["trials"]))
    residual = np.concatenate([s["residual"] for s in stretches])
    order = np.argsort(main["trials"])
    d = main["residual"][order] + main["offset"]
    half = len(d) // 2
    halves = float(np.median(d[:half]) - np.median(d[half:])) if half >= 5 else np.nan

    # ---- 4. The stretches in time order, each with the DAQ time it starts at ------
    # Each trial is labelled with the stretch that paired it; a stretch starts halfway
    # between the last trial of the one before and its own first.
    label = np.full(len(starts), -1)
    for k, s in enumerate(stretches):
        label[s["trials"]] = k
    known = np.flatnonzero(label >= 0)
    seq = label[known]
    in_order, bounds = [seq[0]], [-np.inf]
    for a, b in zip(known[:-1], known[1:]):
        if label[b] != in_order[-1] and label[b] not in in_order:
            in_order.append(label[b])
            bounds.append((starts[a] + starts[b]) / 2)
    segments = []
    for k, lo in zip(in_order, bounds):
        # The first stretch runs from the session's start: no DAQ time.
        from_daq_s = None if not np.isfinite(lo) else float(lo)
        segments.append({"offset_s": stretches[k]["offset"], "from_daq_s": from_daq_s})

    # ---- 5. The result, and the checks ----------------------------------------
    result.update({
        "threshold_g": float(threshold), "n_events": int(len(events)), "n_paired": int(n),
        "fraction_paired": n / len(starts),
        "offset_s": main["offset"],
        "stretches": segments,
        "residual_mad_ms": float(np.median(np.abs(residual)) * 1000),
        "residual_p95_ms": float(np.percentile(np.abs(residual), 95) * 1000),
        "halves_ms": halves * 1000,
    })
    checks = {
        "pairs": n >= MIN_PAIRS,
        "fraction": n / len(starts) >= MIN_FRACTION,
        "residual": np.median(np.abs(residual)) <= MAX_MAD_S,
        "halves": np.isfinite(halves) and abs(halves) <= MAX_HALVES_S,
    }
    result["checks"] = checks
    result["passed"] = all(checks.values())
    result["_plot"] = (t, w, valid, starts, events, stretches, threshold)
    return result


def platform_events(t, w, threshold, hold=HOLD_S):
    """
    Onset + ``hold`` of every stretch at or above ``threshold`` lasting ``hold``.

    ``t`` and ``w`` are arrays of reading times and weights; returns an array of times.
    """
    on = (w >= threshold).astype(np.int8)
    ups = np.flatnonzero(np.diff(on) == 1) + 1
    downs = np.flatnonzero(np.diff(on) == -1) + 1
    out = []
    for up in ups:
        later = downs[downs > up]
        end = t[later[0]] if len(later) else t[-1]
        if end - t[up] >= hold:
            out.append(t[up] + hold)
    return np.asarray(out)


def fit_offsets(starts, events, max_stretches=3):
    """
    One offset per stretch of the session; each with its paired trial indices.

    Returns a list of ``{"offset", "trials", "residual"}`` dicts, in the order found.
    """
    stretches, free = [], np.arange(len(starts))
    for _ in range(max_stretches):
        if len(free) < MIN_PAIRS:
            break
        pairs, line = legacy_trials._pair(list(starts[free]), list(events))
        if line is None:
            break

        # The offset: the median of the paired differences, trimmed to those within
        # TRIM_S of it, three times.
        i = np.array(list(pairs)); j = np.array(list(pairs.values()))
        d = starts[free][i] - events[j]
        for _ in range(3):
            if len(d) == 0:
                break
            m = np.median(d)
            keep = np.abs(d - m) < TRIM_S
            d, i, j = d[keep], i[keep], j[keep]
        if len(d) < MIN_PAIRS:
            break
        offset = float(np.median(d))
        trials = free[i]

        # A later stretch must be a block (see BLOCK_OWN), not pairs scattered by chance.
        if stretches:
            lo, hi = trials.min(), trials.max()
            span = hi - lo + 1
            others = sum(int(((s["trials"] >= lo) & (s["trials"] <= hi)).sum()) for s in stretches)
            if not (len(trials) / span >= BLOCK_OWN and others / span <= BLOCK_OTHERS):
                break

        stretches.append({"offset": offset, "trials": trials, "residual": d - offset})
        free = np.setdiff1d(free, trials)
    return stretches


def _plain(value):
    """JSON for numpy scalars; anything else as its string."""
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def apply(record, result):
    """Write ``scales.csv`` timed by the fit, keeping the untimed one, and stamp ``session.json``."""
    out = record.out_dir
    if not (out / UNPLACED).exists():
        shutil.copy2(out / "scales.csv", out / UNPLACED)
    scales = scales_source.read(record.path("scales_csv"))
    times = reading_times(scales.timestamps, result["stretches"])
    placement = Placement(name="scales", times=times, n_total=len(times),
                          n_unmatched=int(np.isnan(times).sum()), n_pulses=0)
    outputs.scales.write(out, placement, scales)

    # What was done, in session.json.
    path = record.path("session_json")
    document = json.loads(path.read_text())
    document["scales_timing"] = {
        "made_by": BY, "made_at": datetime.now().isoformat(timespec="seconds"),
        "method": "platform events (weight >= threshold for 1 s, at onset + 1 s) paired "
                  "with trial starts; one offset per stretch, slope fixed at 1",
        **{k: v for k, v in result.items() if not k.startswith("_")},
    }
    path.write_text(json.dumps(document, indent=2, default=_plain))


def reading_times(t, stretches):
    """
    Each reading's DAQ time: its own stretch's offset, by where that puts it.

    ``t`` is an array of scales-clock times; ``stretches`` the fit's
    ``{"offset_s", "from_daq_s"}`` list. Unplaced readings are NaN.
    """
    out = np.full(len(t), np.nan)
    valid = t > 0
    for k, s in enumerate(stretches):
        lo = s["from_daq_s"] if s["from_daq_s"] is not None else -np.inf
        hi = stretches[k + 1]["from_daq_s"] if k + 1 < len(stretches) else np.inf
        daq = t + s["offset_s"]
        take = valid & (daq >= lo) & (daq < hi)
        out[take] = daq[take]
    return out


def draw(result, path):
    """Three panels: the offset histogram, residuals over the session, one stretch of weight."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t, w, valid, starts, events, stretches, threshold = result["_plot"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
    diffs = (starts[:, None] - events[None, :]).ravel()
    main = result["offset_s"]
    near = diffs[np.abs(diffs - main) < 3]
    ax[0].hist(near, bins=np.arange(main - 3, main + 3, 0.02), color="#2a78d6")
    ax[0].set(title=f"trial start − platform event (all pairs ±3 s)\noffset {main:.3f} s",
              xlabel="s", ylabel="count")
    for s in stretches:
        tt = starts[s["trials"]]
        ax[1].plot(tt / 60, s["residual"] * 1000, ".", ms=3, label=f"offset {s['offset']:.3f} s")
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set(title=f"residual per paired trial  (MAD {result['residual_mad_ms']:.0f} ms, "
                    f"halves {result['halves_ms']:.0f} ms)", xlabel="session time (min)",
              ylabel="ms", ylim=(-200, 200))
    ax[1].legend(fontsize=7)
    mid = starts[len(starts) // 2]
    times = reading_times(t, result["stretches"])
    win = (times > mid - 30) & (times < mid + 30)
    ax[2].plot(times[win] - mid, w[win], lw=0.8, color="#555")
    ax[2].axhline(threshold, color="#eb6834", lw=0.8, ls="--", label=f"threshold {threshold:.0f} g")
    for s in starts[(starts > mid - 30) & (starts < mid + 30)]:
        ax[2].axvline(s - mid, color="#1baf7a", lw=1)
    ax[2].set(title="weight (placed) with trial starts (green)", xlabel="s from mid-session",
              ylabel="g")
    ax[2].legend(fontsize=7)
    verdict = "APPLIED" if result.get("applied") else f"NOT APPLIED — {result.get('reason', '')}"
    fig.suptitle(f"{result['session']}   threshold {threshold:.0f} g   "
                 f"{result['n_paired']}/{result['n_trials']} trials paired   {verdict}",
                 fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _summary_plot(table, usual, path):
    """Offset per session over time, and the spread of residuals and pairing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = table[table.offset_s.notna()]
    ok = t.applied.astype(bool)
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
    ax[0].plot(pd.to_datetime(t.session.str[:6], format="%y%m%d")[ok], t.offset_s[ok], "o",
               ms=3, color="#2a78d6", label="applied")
    ax[0].plot(pd.to_datetime(t.session.str[:6], format="%y%m%d")[~ok], t.offset_s[~ok], "x",
               ms=4, color="#eb6834", label="not applied")
    ax[0].set(title=f"offset per session (cohort {usual:.2f} s)", ylabel="s",
              ylim=(usual - 4, usual + 4))
    ax[0].legend(fontsize=7)
    ax[0].tick_params(axis="x", rotation=30)
    ax[1].hist(t.residual_mad_ms[ok], bins=30, color="#2a78d6")
    ax[1].set(title="median |residual| per applied session", xlabel="ms")
    ax[2].hist(t.fraction_paired[ok], bins=20, color="#2a78d6")
    ax[2].set(title="fraction of trials paired", xlabel="fraction")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("cohort")
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-dir", help="write the report here instead of the "
                                             "cohort's alignment_report/scales/ (a dry "
                                             "run writes none without it)")
    args = parser.parse_args()
    repair_scales_timing(args.cohort, only=args.only, dry_run=args.dry_run,
                         report_dir=args.report_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
