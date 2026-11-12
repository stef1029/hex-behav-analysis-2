"""
Put the right mouse on legacy sessions whose typed Mouse ID was wrong — a one-off fix.
=====================================================================================

::

    python repair_mouse_id.py <cohort> [--dry-run] [--only ...]

What went wrong
---------------
On the pre-rewrite rigs the mouse ID was typed into the controller by hand, and
sometimes typed wrong — ``M106-1b`` for ``M106-3b``, ``M108-1e`` for ``M108-3e``, a
stray leading comma. The session folder was renamed to the right mouse
afterwards, and every analysis since (the thesis included) has taken the mouse from the
folder name. The legacy converter, though, preferred the typed header, so the
``-metadata.json`` it wrote — and the ``processed/session.json`` and NWBs built from it —
carried the typo, and per-mouse statistics split one animal into two.

What this does
--------------
For each session whose converter-written metadata disagrees with the mouse in its
folder name, it sets ``mouse_id`` to the folder's mouse in the two derived files:

* ``<session>-metadata.json`` — written by ``legacy/convert.py``, not by the rig
* ``processed/session.json`` — written by post-processing from that metadata

and records the typed value beside it as ``mouse_id_as_typed``. The rig's own files
(the behaviour-data JSON, the DAQ file) are never touched, so what was actually typed is
still there. Nothing else is rebuilt, so the video-timing repairs already in
``processed/`` stay as they are. NWBs built before this still hold the old ID: rebuild
them into a new batch.

Only sessions written by the legacy converter are changed — a 2026-pipeline session's
metadata is the rig's own record, and a disagreement there is reported, not fixed.
A summary goes to ``<cohort>/mouse_id_repair/summary.csv``.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Runnable as a file as well as a module, like the other repairs.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from hba2.cohort import Cohort  # noqa: E402


def folder_mouse(session_id):
    """The mouse in a session folder's name: everything after the date and time."""
    return session_id.split("_", 2)[2] if session_id.count("_") >= 2 else ""


def _rewrite(path, mouse, typed, dry_run):
    """Set ``mouse_id`` in one JSON file; True if it was (or, dry run, would be) changed."""
    document = json.loads(path.read_text())
    if document.get("mouse_id") == mouse:
        return False
    if not dry_run:
        document.setdefault("mouse_id_as_typed", typed)
        document["mouse_id"] = mouse
        document.setdefault("mouse_id_repair", {
            "by": "hba2/postprocessing/repairs/repair_mouse_id.py",
            "at": datetime.now().isoformat(timespec="seconds"),
            "why": "the typed Mouse ID disagreed with the session folder's mouse"})
        path.write_text(json.dumps(document, indent=2))
    return True


def repair_mouse_id(cohort, *, only=None, dry_run=False):
    """
    Every session's decision, one row each; the changed ones have ``status == 'fixed'``.

    ``cohort`` is a :class:`Cohort`, or its folder as a Path or a string; ``only`` a
    tuple of session IDs to limit it to.
    """
    cohort = cohort if isinstance(cohort, Cohort) else Cohort(cohort)
    rows = []
    for sid, record in cohort.sessions.items():
        if only and sid not in only:
            continue
        meta_path = record.raw.get("metadata")
        if meta_path is None or not Path(meta_path).exists():
            continue
        meta = json.loads(Path(meta_path).read_text())
        mouse, typed = folder_mouse(sid), str(meta.get("mouse_id", ""))
        row = {"session_id": sid, "folder_mouse": mouse, "typed": typed}
        if not mouse or typed == mouse:
            rows.append({**row, "status": "agrees"})
            continue
        if "converted_from" not in meta:
            rows.append({**row, "status": "reported only: rig-written metadata"})
            continue
        changed = [_rewrite(Path(meta_path), mouse, typed, dry_run)]
        session_json = record.out_dir / "session.json"
        if session_json.exists():
            changed.append(_rewrite(session_json, mouse, typed, dry_run))
        # Already right in both files; or would be changed (dry run); or changed.
        if not any(changed):
            status = "already fixed"
        elif dry_run:
            status = "would fix"
        else:
            status = "fixed"
        rows.append({**row, "status": status})

    table = pd.DataFrame(rows)
    if not dry_run and not table.empty:
        out = Path(cohort.base) / "mouse_id_repair"
        out.mkdir(exist_ok=True)
        table[table["status"] != "agrees"].to_csv(out / "summary.csv", index=False)
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("cohort")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args(argv)
    table = repair_mouse_id(args.cohort, only=tuple(args.only) if args.only else None,
                            dry_run=args.dry_run)
    print(table["status"].value_counts().to_string())
    print(table[table["status"] != "agrees"].to_string(index=False))


if __name__ == "__main__":
    main()
