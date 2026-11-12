#!/usr/bin/env python3
"""
What follows DeepLabCut for a cohort: pose, the session index, NWB.
===================================================================

Run once DeepLabCut has finished — normally by Slurm, as the job
:func:`run_dlc_pipeline.submit_after_dlc` makes wait on the cohort's arrays::

    python after_dlc.py <cohort> --mouse-model T1=<config.yaml> [--mouse-model ...]

Mice are given with the same config paths DeepLabCut was run with, and each model is
named by its Task as everywhere else. Per model it builds pose for that model's mice;
then the session index is refreshed, since NWB takes its labels from it; then the
listed mice's sessions are added to an NWB batch.

**Which batch.** ``--batch`` names one; without it, the cohort's newest. A batch that
exists is topped up with its own recipe's settings — its model, sort and curation — so
every file in it stays built the same way; a session that batch's model has no output
for is skipped and says so. A batch that does not exist yet is started, with each listed
mouse's model, and is named ``<year>-<month>_v1`` when no name was given.

**What succeeded is built; what did not is skipped, with its reason.** A session
DeepLabCut failed on has no output, so pose and NWB skip it and every other session is
still built. A stage that fails outright — a missing port table, say — is logged and
the remaining stages run anyway. Every stage also skips what is already current, so a
second run costs a scan, not a rebuild.

Exits non-zero only when a stage raised, or when sessions errored and none were built,
so a Slurm ``FAILED`` means something is really wrong rather than one bad video.
"""

import argparse
import sys
import traceback
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hba2 import Cohort, build_index, build_nwb, build_pose  # noqa: E402
from hba2.pose.run_dlc_pipeline import (  # noqa: E402
    parse_mouse_models, resolve_configs,
)


def _log(message):
    print(message, flush=True)


def build_after_dlc(cohort, mouse_models, *, batch=None, reprocess=False):
    """
    Pose, index and NWB for every listed mouse's sessions. True when nothing failed
    badly enough to report as a failed job — see the module docstring.

    ``cohort`` is the cohort folder, a Path or a string. ``mouse_models`` is
    ``{mouse: config.yaml path}``. ``batch``: the NWB batch to add them to. None means
    the cohort's newest, or a new one if it has none.
    """
    base = Path(cohort)
    tasks = resolve_configs(mouse_models)            # {config: Task}
    by_task = {}                                     # {Task: set of mice}
    for mouse, config in mouse_models.items():
        by_task.setdefault(tasks[config], set()).add(mouse)

    healthy = True

    # A fresh Cohort for each stage: each one reads what the stage before it wrote.
    # ---- 1. Pose, per model ---------------------------------------------
    for task, mice in sorted(by_task.items()):
        _log(f"\n=== Pose · {task} · {', '.join(sorted(mice))} ===")
        try:
            result = build_pose(Cohort(base), task, reprocess=reprocess, log_fn=_log)
            _log(result.describe())
            if result.failed and not result.ok:
                healthy = False
        except Exception:
            _log(traceback.format_exc())
            healthy = False

    # ---- 2. The session index, which NWB takes its labels from -------------
    _log("\n=== Session index ===")
    try:
        build_index(Cohort(base), log_fn=_log)
    except Exception:
        _log(traceback.format_exc())
        healthy = False

    # ---- 3. NWB, for the listed mice ----------------------------------------
    try:
        cohort = Cohort(base)
        if batch is None:
            batch = cohort.batch

        # An existing batch keeps its own settings; a new one takes each mouse's model.
        if batch in cohort.batches:
            settings = dict(Cohort(base, batch=batch).recipe.get("parameters", {}))
        else:
            if batch is None:
                batch = f"{date.today():%Y-%m}_v1"
            model = {mouse: tasks[config] for mouse, config in mouse_models.items()}
            settings = {"model": model}
        _log(f"\n=== NWB · batch {batch} · {', '.join(sorted(mouse_models))} ===")

        # Just the listed mice: the others' sessions have no new output, and listing
        # each as skipped would bury the ones that matter.
        only = tuple(r.session_id for r in cohort if r.mouse_id in mouse_models)
        if not only:
            _log("  no sessions for these mice")
        else:
            result = build_nwb(cohort, batch=batch, only=only, reprocess=reprocess,
                               log_fn=_log, **settings)
            failed = [r for r in result.results if r.error]
            if failed and not result.ok:
                healthy = False
    except Exception:
        _log(traceback.format_exc())
        healthy = False

    return healthy


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Pose, session index and NWB for a cohort, after DeepLabCut.")
    parser.add_argument("cohort", help="cohort folder")
    parser.add_argument("--mouse-model", action="append", required=True,
                        metavar="MOUSE=CONFIG",
                        help="a mouse and the DeepLabCut config.yaml it was analysed "
                             "with; repeat per mouse")
    parser.add_argument("--batch", help="the NWB batch to add the sessions to "
                                        "(default: the cohort's newest, or a new one)")
    parser.add_argument("--reprocess", action="store_true",
                        help="rebuild even where the output is still current")
    args = parser.parse_args(argv)

    ok = build_after_dlc(args.cohort, parse_mouse_models(args.mouse_model),
                         batch=args.batch, reprocess=args.reprocess)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
