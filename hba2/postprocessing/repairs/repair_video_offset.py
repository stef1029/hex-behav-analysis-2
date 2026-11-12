"""
Remove constant offsets between a session's video and its DAQ clock — a one-off fix.
===================================================================================

**Run after post-processing and after** ``repair_video_timing``::

    python repair_video_offset.py <cohort> <model task> [--dry-run] [--only ...]

What went wrong
---------------
In some sessions recorded before the 2026 pipeline (about 8–18% of sessions on the
cohorts audited on 2026-10-15), the video sits a constant 0.1–5 s away from the DAQ
clock — sometimes with one jump partway through the session. The cause was in the
recording system of the time and is not recoverable; the system has since changed and
does not do it. What shows it is the same ground truth the timing repair uses: when a
port's beam breaks the head is at that port. In an affected session, the head is away
from the touched port at almost every touch under the recorded timing, and at it once the
video is moved by a few seconds.

What this does
--------------
For each session, from the touches alone:

1. **One offset for the whole session**: the shift of the video against the touches
   that puts the head at the touched port most often (a coarse search over ±120 s, then
   frame by frame around the best).
2. **Or two, with one jump**: where one offset leaves the session badly placed, the best
   split into two stretches, each with its own offset.
3. **Applied only when all of these hold** — otherwise the session is left exactly as it
   was, and the reason recorded:

   * enough touches to fit (``MIN_TOUCHES``; ``MIN_SEGMENT`` per stretch);
   * the shift is real: at least ``MIN_SHIFT_S`` (in one stretch at least — a stretch
     before a jump may rightly need none), and the touches it moves onto the port
     significantly outnumber those it moves off (one-sided binomial, p < ``ACCEPT_P``,
     the timing repair's own test);
   * it matters: the share of touches with the head far from the port falls by at least
     ``MIN_GAIN``;
   * it is constant: fitted on each half of a stretch separately, the offsets agree to
     within ``CONSTANT_S``.

``frame_times.csv`` is rewritten with each frame's time moved by its stretch's offset.
**No frame is moved to another frame's time and no time is invented**: across a jump, a
frame whose moved time would fall out of order is left with no time (it overlapped the
other stretch), which the rest of the pipeline already treats as an untimed frame.

What it writes, and what it never touches
------------------------------------------
::

    <session>/processed/
    ├── frame_times.csv               moved, if applied
    ├── frame_times_before_offset.csv the file as it was, copied first
    └── video_offset.json             the fit, the evidence and the decision

    <cohort>/video_offset/summary.csv every session decided

Ports are each session's own — the ones its pose was built with (``pose.json``) — so a
cohort calibrated differently from the shipped rig table is judged in its own geometry.
Nothing else is changed. Pose notices the new ``frame_times.csv`` and rebuilds.

A session already offset is skipped (``--force`` decides again from
``frame_times_before_offset.csv``). Post-processing re-run on a session writes a fresh
``frame_times.csv`` and undoes this, as it does the timing repair — run both again after.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from hba2.cohort import Cohort  # noqa: E402
from hba2.pose.pose import (  # noqa: E402
    _readable_json, dlc_from_table, head_bearing, head_position, read_dlc_table)
from hba2.pose.frame_lookup import frames_at  # noqa: E402
from hba2.postprocessing.repairs.repair_video_timing import FAR_PX, _distances  # noqa: E402

FRAME_TIMES = "frame_times.csv"
BEFORE_CSV = "frame_times_before_offset.csv"
OFFSET_JSON = "video_offset.json"
SUMMARY_DIR = "video_offset"
BY = "hba2/postprocessing/repairs/repair_video_offset.py"

FPS = 30.0
COARSE = np.arange(-120.0, 120.01, 0.5)
MIN_TOUCHES = 40
MIN_SEGMENT = 25
MIN_SHIFT_S = 0.1
MIN_GAIN = 0.10
ACCEPT_P = 0.001
CONSTANT_S = 0.3
#: A second stretch is used only if it places the head this much better again.
SPLIT_GAIN = 0.05
SPLITS = np.arange(0.15, 0.851, 0.05)


def repair_video_offset(cohort, model, *, only=None, dry_run=False, force=False, workers=8):
    """
    Decide every session in a cohort; returns one row per session.

    ``cohort`` is a :class:`Cohort`, or its folder as a Path or a string; ``only`` a
    tuple of session IDs to limit it to. A dry run decides and writes nothing.
    """
    cohort = cohort if isinstance(cohort, Cohort) else Cohort(cohort)
    records = [r for r in cohort if not only or r.session_id in only]
    job = partial(offset_session, model=model, dry_run=dry_run, force=force)
    with ProcessPoolExecutor(workers) as pool:
        table = pd.DataFrame([asdict(r) for r in pool.map(job, records)])
    if not dry_run and not table.empty:
        folder = cohort.base / SUMMARY_DIR
        folder.mkdir(exist_ok=True)
        summary = folder / "summary.csv"
        # Earlier decisions are kept, except for the sessions decided again now.
        old = pd.read_csv(summary) if summary.exists() else pd.DataFrame()
        if len(old):
            old = old[~old["session_id"].isin(table["session_id"])]
        merged = pd.concat([old, table.assign(model=model)], ignore_index=True)
        merged.to_csv(summary, index=False)
    return table


def offset_session(record, model, *, dry_run=False, force=False):
    """Decide one session, and unless a dry run, record the decision and apply it."""
    result = OffsetResult(session_id=record.session_id)
    try:
        # ---- 1. Whether there is anything to decide with ----------------
        out = record.out_dir
        if not record.is_processed or not (out / FRAME_TIMES).exists():
            result.status, result.reason = "skipped", "not post-processed"
            return result
        if (out / OFFSET_JSON).exists() and not force:
            result.status, result.reason = "skipped", "already decided"
            return result
        key = record.dlc_key(model)
        if key is None or not record.dlc[key].get("csv"):
            result.status, result.reason = "skipped", f"no DeepLabCut output for '{model}'"
            return result
        pose_json = record.directory / "dlc" / key / "pose.json"
        if not pose_json.exists():
            result.status, result.reason = "skipped", "no pose.json (no ports to judge with)"
            return result

        # ---- 2. Read: ports, frame times, the head, the touches ---------
        ports = np.asarray(json.loads(pose_json.read_text())["ports"]["coords"], dtype=float)
        # Forced, the decision is made again from the file as it was before any offset.
        source = FRAME_TIMES
        if force and (out / BEFORE_CSV).exists():
            source = BEFORE_CSV
        current = pd.read_csv(out / source)
        dlc = dlc_from_table(read_dlc_table(record.dlc[key]["csv"]))
        bearing, method = head_bearing(dlc)
        position = head_position(dlc, bearing, method)
        events = pd.read_csv(record.path("events"))
        touched = events[(events["event"] == "sensor_touch") & (events["index"] >= 0)]
        touches = _Touches(current, position, ports, touched["t"].to_numpy(float),
                           touched["index"].to_numpy(int))
        if len(touches.t) < MIN_TOUCHES:
            result.status, result.reason = "skipped", f"only {len(touches.t)} touches"
            return result

        # ---- 3. Fit, and decide ----------------------------------------
        plan = fit_offsets(touches)
        apply, reason, evidence = judge(touches, plan)
        result.offsets_s = [round(x, 3) for x in plan["offsets_s"]]
        result.far_before, result.far_after = evidence["far_before"], evidence["far_after"]
        no_shift = all(abs(x) < MIN_SHIFT_S for x in plan["offsets_s"])
        if apply:
            result.status = "applied"
        elif no_shift:
            result.status = "not_needed"
        else:
            result.status = "rejected"
        # An accepted plan has no reasons against it: say what it did instead.
        result.reason = reason or (f"{len(plan['offsets_s'])} stretch(es); head far from port "
                                   f"{evidence['far_before']:.1%} → {evidence['far_after']:.1%}")
        if dry_run:
            return result

        # ---- 4. Write: the original first, then the decision, then the moved file
        manifest = {"session_id": record.session_id, "status": result.status,
                    "reason": result.reason, "decided_at": datetime.now().isoformat(timespec="seconds"),
                    "by": BY, "model": key, "ports_from": str(pose_json),
                    "split_at_s": plan["split_at_s"], "evidence": evidence,
                    "frame_times_original": BEFORE_CSV if apply else None}
        if apply and not (out / BEFORE_CSV).exists():
            shutil.copy2(out / FRAME_TIMES, out / BEFORE_CSV)
        (out / OFFSET_JSON).write_text(_readable_json(manifest))
        if apply:
            moved_frame_times(current, plan).to_csv(out / FRAME_TIMES, index=False,
                                                    float_format="%.6f")
    except Exception as exc:                       # noqa: BLE001 — reported, not raised
        result.status, result.reason = "error", f"{type(exc).__name__}: {exc}"
    return result


class _Touches:
    """Everything needed to ask: with the video moved by x, how far was the head?"""

    def __init__(self, frame_times, position, ports, touch_t, touch_port):
        self.ft, self.pos, self.ports = frame_times, position, ports
        self.t, self.port = touch_t, touch_port

    def distances(self, idx, shift):
        return _distances(frames_at(self.ft, self.t[idx] + shift, len(self.pos)),
                          self.pos, self.ports, self.port[idx])

    def far(self, idx, shift):
        d = self.distances(idx, shift)
        ok = np.isfinite(d)
        return float(np.mean(d[ok] > FAR_PX)) if ok.sum() >= 10 else np.nan

    def fit(self, idx, grid=COARSE):
        """Best shift for these touches: coarse by far share, then frame by frame."""
        coarse = np.array([self.far(idx, x) for x in grid])
        if np.all(np.isnan(coarse)):
            return np.nan
        best = grid[np.nanargmin(coarse)]
        fine = best + np.arange(-1.0, 1.0001, 1.0 / FPS)
        medians = []
        for x in fine:
            d = self.distances(idx, x)
            medians.append(np.nanmedian(d) if np.isfinite(d).sum() >= 10 else np.nan)
        if np.all(np.isnan(medians)):
            return float(best)
        return float(fine[np.nanargmin(medians)])


def fit_offsets(touches):
    """One offset, or two with a jump, and the evidence for each."""
    everything = np.arange(len(touches.t))
    single = touches.fit(everything)
    plan = {"split_at_s": None, "offsets_s": [single], "segments": [everything]}
    far_single = touches.far(everything, single)

    # A second stretch when one offset leaves the session badly placed, or is not the
    # same in both halves of the session (a jump partway through).
    halves = np.array_split(everything, 2)
    drift = [touches.fit(h, single + np.arange(-3.0, 3.01, 0.5)) for h in halves]
    unsteady = any(np.isnan(drift)) or abs(drift[0] - drift[1]) > CONSTANT_S
    if far_single > MIN_GAIN or unsteady:
        best = None
        for share in SPLITS:
            k = int(round(share * len(everything)))
            if k < MIN_SEGMENT or len(everything) - k < MIN_SEGMENT:
                continue
            a, b = everything[:k], everything[k:]
            xa, xb = touches.fit(a, COARSE[::2]), touches.fit(b, COARSE[::2])
            if np.isnan(xa) or np.isnan(xb):
                continue
            far = (touches.far(a, xa) * len(a) + touches.far(b, xb) * len(b)) / len(everything)
            if best is None or far < best[0]:
                best = (far, k, xa, xb)
        if (best is not None and (best[0] <= far_single - SPLIT_GAIN or unsteady)
                and abs(best[3] - best[2]) >= CONSTANT_S):
            _, k, xa, xb = best
            split = float((touches.t[k - 1] + touches.t[k]) / 2)
            plan = {"split_at_s": split, "offsets_s": [xa, xb],
                    "segments": [everything[:k], everything[k:]]}
    return plan


def _far_share(d):
    """The share of measured distances over ``FAR_PX``; NaN if none was measured."""
    if not np.isfinite(d).any():
        return np.nan
    return float(np.mean(d[np.isfinite(d)] > FAR_PX))


def judge(touches, plan):
    """Whether to apply the plan, why, and the numbers behind it: ``(apply, reason, evidence)``."""
    evidence = {"stretches": []}
    before_d = touches.distances(np.arange(len(touches.t)), 0.0)
    after_d = np.full(len(touches.t), np.nan)
    reasons = []
    for idx, x in zip(plan["segments"], plan["offsets_s"]):
        halves = np.array_split(idx, 2)
        xs = [touches.fit(h, x + np.arange(-3.0, 3.01, 0.5)) for h in halves]
        after_d[idx] = touches.distances(idx, x)
        stretch = {"touches": int(len(idx)), "offset_s": round(x, 3),
                   "offset_frames": round(x * FPS, 1),
                   "half_offsets_s": [round(v, 3) for v in xs]}
        evidence["stretches"].append(stretch)
        if len(idx) < MIN_SEGMENT:
            reasons.append(f"a stretch has only {len(idx)} touches")
        if any(np.isnan(xs)) or abs(xs[0] - xs[1]) > CONSTANT_S:
            reasons.append(f"not constant: halves fit {xs[0]:+.2f} s and {xs[1]:+.2f} s")

    both = np.isfinite(before_d) & np.isfinite(after_d)
    fixed = int((both & (before_d > FAR_PX) & (after_d <= FAR_PX)).sum())
    broken = int((both & (before_d <= FAR_PX) & (after_d > FAR_PX)).sum())
    p = 1.0
    if fixed + broken:
        p = binomtest(fixed, fixed + broken, 0.5, alternative="greater").pvalue
    evidence.update(touches=int(len(touches.t)), far_before=round(_far_share(before_d), 4),
                    far_after=round(_far_share(after_d), 4), touches_fixed=fixed,
                    touches_broken=broken, p_better=float(f"{p:.3g}"),
                    median_px_before=round(float(np.nanmedian(before_d)), 1),
                    median_px_after=round(float(np.nanmedian(after_d)), 1))
    if all(abs(x) < MIN_SHIFT_S for x in plan["offsets_s"]):
        reasons.append(f"no stretch is offset by {MIN_SHIFT_S} s or more")
    if len(touches.t) < MIN_TOUCHES:
        reasons.append(f"only {len(touches.t)} touches")
    if p >= ACCEPT_P:
        reasons.append(f"not significantly better (p = {p:.3g})")
    if not evidence["far_before"] - evidence["far_after"] >= MIN_GAIN:
        reasons.append(f"far share only {evidence['far_before']:.1%} → {evidence['far_after']:.1%}")
    return not reasons, "; ".join(dict.fromkeys(reasons)), evidence


def moved_frame_times(frame_times, plan):
    """Each frame's time moved by its stretch's offset; overlapping frames left untimed."""
    t = frame_times["t"].to_numpy(dtype=float)
    if plan["split_at_s"] is None:
        new = t - plan["offsets_s"][0]
    else:
        x1, x2 = plan["offsets_s"]
        t1, t2 = t - x1, t - x2
        first = t1 < plan["split_at_s"]
        second = t2 >= plan["split_at_s"]
        new = np.where(first & ~second, t1, np.where(second & ~first, t2, np.nan))
    # Keep the times in order: a frame not later than every frame before it overlapped.
    valid = np.isfinite(new)
    running = np.maximum.accumulate(np.where(valid, new, -np.inf))
    previous = np.concatenate(([-np.inf], running[:-1]))
    new[valid & (new <= previous)] = np.nan
    out = frame_times.copy()
    out["t"] = new
    return out


@dataclass
class OffsetResult:
    """What was decided for one session."""

    session_id: str
    status: str = ""          # applied · not_needed · rejected · skipped · error
    reason: str = ""
    offsets_s: list | None = None
    far_before: float | None = None
    far_after: float | None = None


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("cohort")
    parser.add_argument("model", help="DeepLabCut model (its Task) whose pose and ports judge it")
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    table = repair_video_offset(args.cohort, args.model, only=tuple(args.only) if args.only else None,
                                dry_run=args.dry_run, force=args.force, workers=args.workers)
    print(table["status"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
