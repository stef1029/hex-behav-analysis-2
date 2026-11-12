"""
A whole legacy cohort, from raw folders to NWB files.
====================================================

Two steps, each safe to re-run:

1. :func:`convert_cohort` — every session folder through :func:`.convert.convert_session`,
   in parallel. Writes ``legacy_conversion_report.txt`` in the cohort folder.
2. :func:`finish_cohort` — what any cohort gets after post-processing, from the
   package's own stages: port calibration, pose, the session index, NWB files.

Ports are calibrated **per cohort**, into ``<cohort>/port_coordinates/`` rather than
the package's shared table. Old rig numbers mean different hardware from new ones (old
1 and 2 were the red rigs; old 3 is not traceable to any current rig), and even one old
rig's camera moved between cohorts, so each cohort measures its own from where the head
is when each beam breaks — the same method, kept apart.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .convert import convert_session, find_sessions

#: What each old rig number was. Old numbers are kept as recorded; the name says which
#: physical rig they mean, so an old rig 1 is never read as a current Rig 1.
RIG_NAMES = {1: "Red rig 1", 2: "Red rig 2", 3: "Rig 3 (pre-2026)"}

#: Mouse IDs that were rig tests, not animals: ``include`` is set false for them.
TEST_MOUSE = re.compile(r"^test\d*$", re.IGNORECASE)

REPORT = "legacy_conversion_report.txt"

#: What a port needs to anchor the hexagon that places a rig's unmeasured ports.
MIN_ANCHOR_TOUCHES = 100
MAX_ANCHOR_SCATTER_PX = 50.0


# =====================================================================
# 1. Convert
# =====================================================================

def convert_cohort(base, *, nonspatial_cues=(), rig_names=RIG_NAMES, overwrite=False,
                   workers=16, log=print):
    """
    Every legacy session folder under ``base`` (a Path or a string) through
    :func:`.convert.convert_session`, in ``workers`` processes, then
    ``legacy_conversion_report.txt`` in the cohort folder.

    Args:
        nonspatial_cues: a tuple of channels that cued a trial without naming a port —
            see :func:`.trials.detect`.
        rig_names: what each old rig number was (:data:`RIG_NAMES`).
        overwrite: redo sessions already converted.
        log: called with each line of progress.

    Returns the list of :class:`~.convert.ConvertResult`, by session ID.
    """
    base = Path(base)
    sessions = find_sessions(base)
    log(f"{base.name}: {len(sessions)} legacy session folder(s)")

    # ---- 1. Every session, in parallel -------------------------------------
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(convert_session, s, rig_names=rig_names,
                               nonspatial_cues=nonspatial_cues, overwrite=overwrite): s
                   for s in sessions}
        for number, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if number % 25 == 0 or not result.ok:
                line = f"  [{number}/{len(sessions)}] {result.session_id}"
                if not result.ok:
                    line += f": FAILED — {result.error}"
                log(line)
    results.sort(key=lambda r: r.session_id)

    # ---- 2. The report -----------------------------------------------------
    ok = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]
    lines = [f"{base.name} — legacy conversion, {datetime.now():%Y-%m-%d %H:%M}",
             f"{len(ok)} converted, {len(failed)} failed, "
             f"{sum(r.n_trials for r in ok):,} trials", ""]
    if failed:
        lines += ["FAILED"] + [f"  {r.session_id}  {r.error}" for r in failed] + [""]
    noted = []
    for r in ok:
        for w in r.warnings:
            if w != "already converted" and not w.startswith("Traceback"):
                noted.append((r.session_id, w))
    if noted:
        lines += ["WARNINGS"] + [f"  {sid}  {w}" for sid, w in noted]
    (base / REPORT).write_text("\n".join(lines) + "\n", encoding="utf-8")

    log(f"{base.name}: {len(ok)} converted, {len(results) - len(ok)} failed — "
        f"see {REPORT}")
    return results


# =====================================================================
# 2. Finish: ports, pose, index, NWB
# =====================================================================

def finish_cohort(base, models, *, batch, sort=None, curation=None, workers=8, log=print):
    """
    Calibrate, build pose for each model, index, and build the batch.

    Args:
        base: the cohort folder, a Path or a string.
        models: a list of DeepLabCut models in order of preference — most recent first.
            Each session's NWB uses the first it has. Ports are calibrated from every
            model, the one covering most sessions taking precedence.
        batch, sort, curation: the NWB batch to build, and the spike sort and curation
            it takes units from (None for none).
        workers: processes for pose and NWB.
        log: called with each line of progress.
    """
    from ..nwb.build_nwb import build_nwb
    from ..cohort import Cohort
    from ..pose.calibrate_ports import calibrate
    from ..pose.pose import approved_ports
    from ..cohort.session_index import build_index, index_path

    base = Path(base)
    cohort = Cohort(base)
    port_dir = base / "port_coordinates"

    # ---- 1. Ports, measured from this cohort alone ---------------------------
    # Every model, the best-covered last: calibrate() replaces the rigs it measures and
    # keeps the rest, so a rig only one model saw still gets ports, and a rig several
    # saw takes them from the model with the most sessions.
    tables = approved_ports(port_dir / "port_coordinates.json")
    if not tables:
        coverage = {}
        for model in models:
            coverage[model] = sum(1 for r in cohort if r.is_processed and r.dlc_key(model))
        incomplete = {}
        for model in sorted(models, key=coverage.get):
            if coverage[model]:
                for rig in calibrate(cohort, model, out_dir=port_dir, log=log):
                    if not rig.complete:
                        incomplete[rig.rig] = (rig, model)
        _complete_by_hexagon(port_dir / "port_coordinates.json", incomplete, log)
        tables = approved_ports(port_dir / "port_coordinates.json")
    if not tables:
        log(f"{base.name}: no port calibration possible — stopping before pose")
        return

    # ---- 2. Pose, every model, sessions spread over processes -----------------
    # build_pose is serial by design; each worker runs it on its own share of sessions.
    for model in models:
        ids = [r.session_id for r in cohort if r.is_processed and r.dlc_key(model)]
        if not ids:
            continue
        shares = [tuple(ids[i::workers]) for i in range(min(workers, len(ids)))]
        built = failed = 0
        with ProcessPoolExecutor(max_workers=len(shares)) as pool:
            for result in pool.map(_pose_share, [(str(base), model, tables, share)
                                                 for share in shares]):
                built += len(result.ok)
                failed += len(result.failed)
                for r in result.failed:
                    log(f"  pose {r.session_id}: {r.error}")
        log(f"{base.name}: pose '{model}' — {built} built, {failed} failed")
        cohort = Cohort(base)

    # ---- 3. The index, with test mice marked out ------------------------------
    build_index(cohort, log_fn=log)
    path = index_path(cohort)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    mouse_column = "_mouse_id" if "_mouse_id" in frame else "mouse_id"
    if mouse_column in frame:
        test = frame[mouse_column].str.match(TEST_MOUSE)
        frame.loc[test & (frame["include"] == ""), "include"] = "False"
        frame.loc[test & (frame["exclude_reason"] == ""), "exclude_reason"] = "test mouse"
        frame.to_csv(path, index=False)
        log(f"{path.name}: {int(test.sum())} test-mouse session(s) marked include=False")

    # ---- 4. NWB -----------------------------------------------------------------
    result = build_nwb(Cohort(base), {"*": list(models)}, batch=batch, sort=sort,
                       curation=curation, workers=workers, log_fn=log)
    log(result.describe())


def _complete_by_hexagon(path, incomplete, log):
    """
    Ports a rig's touches never located, placed by the hexagon the others make.

    On rig 3 in spring 2025 the DAQ did not record ports 1 and 2, so no touch says where
    they are. The ports sit on a regular hexagon, so four measured ports fix its centre,
    radius and rotation, and the missing two follow. Only for rigs with at least three
    measured ports and none already in the table; the entry says which were fitted.

    ``path`` is the cohort's ``port_coordinates.json``; ``incomplete`` maps rig number
    -> (that rig's calibration, the model it came from).
    """
    from scipy.optimize import least_squares

    table = json.loads(path.read_text()) if path.exists() else {}
    for number, (rig, model) in incomplete.items():
        key = f"rig_{number}"
        # A port anchors the fit only if its touches really locate it: a blind port can
        # still collect a few stray beam breaks, scattered across the arena.
        scatter = np.asarray(getattr(rig, "scatter", np.zeros(len(rig.coords))), dtype=float)
        known = (np.isfinite(rig.coords).all(axis=1)
                 & (np.asarray(rig.counts) >= MIN_ANCHOR_TOUCHES)
                 & (np.nan_to_num(scatter, nan=np.inf) <= MAX_ANCHOR_SCATTER_PX))
        if key in table or known.sum() < 3:
            continue

        # The hexagon through the measured ports, numbered either way round.
        ports = np.flatnonzero(known)
        best = None
        for sense in (1, -1):
            def residual(x, sense=sense):
                cx, cy, r, theta = x
                angle = theta + sense * ports * np.pi / 3
                return np.concatenate([cx + r * np.cos(angle) - rig.coords[ports, 0],
                                       cy + r * np.sin(angle) - rig.coords[ports, 1]])
            centre = rig.coords[ports].mean(axis=0)
            start = rig.coords[ports[0]] - centre
            fit = least_squares(residual, [*centre, np.hypot(*start),
                                           np.arctan2(start[1], start[0])
                                           - sense * ports[0] * np.pi / 3])
            if best is None or fit.cost < best[0].cost:
                best = (fit, sense)
        fit, sense = best
        cx, cy, r, theta = fit.x

        # The rest placed on it.
        coords = rig.coords.copy()
        for port in np.flatnonzero(~known):
            angle = theta + sense * port * np.pi / 3
            coords[port] = (cx + r * np.cos(angle), cy + r * np.sin(angle))
        rms = float(np.sqrt(np.mean(fit.fun ** 2)))
        fitted = [int(p) for p in np.flatnonzero(~known)]
        table[key] = {
            "coords": np.round(coords, 1).tolist(),
            "derived_from": {"model": model, "sessions": rig.n_sessions,
                             "touches_per_port": rig.counts.tolist(),
                             "fitted_ports": fitted,
                             "hexagon_fit_rms_px": round(rms, 1),
                             "method": "measured ports from touches; the rest placed on "
                                       "the regular hexagon fitted to them"},
        }
        log(f"rig {number}: port(s) {np.flatnonzero(~known).tolist()} placed by hexagon "
            f"fit to the measured ones (rms {rms:.1f} px)")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(sorted(table.items())), indent=2))


def _pose_share(args):
    """One worker's share of a pose build — module level, so it can be pickled."""
    from ..pose.pose import build_pose
    base, model, tables, share = args
    return build_pose(base, model, ports=tables, only=share)
