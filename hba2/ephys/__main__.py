"""
The ephys pipeline from the command line — all of it::

    python -m hba2.ephys run <cohort> --curation C [--sort S]
        [--freeze all | --freeze SESSION ...] [--tracks tracks_all.csv --matching M --level L]
        [--only SESSION ...] [--no-sync] [--reexport]

or one step at a time::

    python -m hba2.ephys adopt  <cohort> <session>... [--old DIR] [--name SORT]
    python -m hba2.ephys freeze <cohort> <sort> <curation> <session>... [--note TEXT]
    python -m hba2.ephys export <cohort> <sort> <curation> <session>...
    python -m hba2.ephys names  <cohort> <tracks.csv> --sort S --curation C \\
                                                --matching NAME --level LEVEL
    python -m hba2.ephys neurons <cohort> --sort S --curation C

``<session>`` may be ``all``: every session that has what the step needs. Each session
is done independently and a failure is reported without stopping the rest.
"""

import argparse
import json
import sys

from ..cohort import Cohort
from ..cohort.neuron_index import build_neuron_index
from .adopt import DEFAULT_NAME, DEFAULT_OLD, adopt_sort
from .curation import freeze
from .export import export
from .names import write_names
from .pipeline import process_ephys_cohort


def _sessions(cohort, asked, eligible):
    """
    The sessions named in ``asked`` (a list of session IDs), or with ``["all"]`` every
    one the function ``eligible`` accepts. A list of records.
    """
    if asked == ["all"]:
        return [r for r in cohort.with_files() if eligible(r)]
    return [cohort.session(session_id) for session_id in asked]


def main():
    """Run the step named on the command line. Returns the exit code: 0, or 1 on failure."""
    parser = argparse.ArgumentParser(prog="python -m hba2.ephys",
                                     description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    steps = parser.add_subparsers(dest="step", required=True)

    adopt = steps.add_parser("adopt", help="move a pipeline sort into ephys_sorts/<name>/")
    adopt.add_argument("cohort")
    adopt.add_argument("sessions", nargs="+")
    adopt.add_argument("--old", default=DEFAULT_OLD)
    adopt.add_argument("--name", default=DEFAULT_NAME)

    frozen = steps.add_parser("freeze", help="copy Phy's current state into a curation")
    frozen.add_argument("cohort")
    frozen.add_argument("sort")
    frozen.add_argument("curation")
    frozen.add_argument("sessions", nargs="+")
    frozen.add_argument("--note", default="")

    exported = steps.add_parser("export", help="a curation's good units and spikes")
    exported.add_argument("cohort")
    exported.add_argument("sort")
    exported.add_argument("curation")
    exported.add_argument("sessions", nargs="+")

    named = steps.add_parser("names", help="neuron names from a matching run")
    named.add_argument("cohort")
    named.add_argument("tracks")
    for option in ("--sort", "--curation", "--matching", "--level"):
        named.add_argument(option, required=True)

    run = steps.add_parser("run", help="the whole pipeline on a cohort, doing what is missing")
    run.add_argument("cohort")
    run.add_argument("--curation", required=True)
    run.add_argument("--sort", default=DEFAULT_NAME)
    run.add_argument("--freeze", nargs="+", default=[],
                     help="sessions whose Phy labels are this curation, or 'all'")
    run.add_argument("--tracks", help="cross-session matching table")
    run.add_argument("--matching", help="with --tracks: a name for that run")
    run.add_argument("--level", help="with --tracks: which of its settings made the tracks")
    run.add_argument("--only", nargs="+", help="just these sessions")
    run.add_argument("--no-sync", dest="sync", action="store_false")
    run.add_argument("--reexport", action="store_true")
    run.add_argument("--old", default=DEFAULT_OLD)

    neurons = steps.add_parser("neurons", help="build or refresh the cohort's neurons.csv")
    neurons.add_argument("cohort")
    neurons.add_argument("--sort", required=True)
    neurons.add_argument("--curation", required=True)

    args = parser.parse_args()
    cohort = Cohort(args.cohort)

    # ---- The whole pipeline --------------------------------------------
    if args.step == "run":
        if args.freeze == ["all"]:
            to_freeze = True
        else:
            to_freeze = args.freeze
        result = process_ephys_cohort(
            cohort, curation=args.curation, sort=args.sort, freeze=to_freeze,
            tracks=args.tracks, matching=args.matching, level=args.level,
            sync=args.sync, old=args.old, only=args.only, reexport=args.reexport,
            log_fn=lambda line: print(line, flush=True))
        return 1 if result.error or any(r.error for r in result.results) else 0

    # ---- The cohort-wide steps -----------------------------------------
    if args.step == "neurons":
        print(build_neuron_index(cohort, sort=args.sort, curation=args.curation).describe())
        return 0

    if args.step == "names":
        report = write_names(cohort, args.tracks, sort=args.sort, curation=args.curation,
                             matching=args.matching, level=args.level)
        print(json.dumps(report, indent=2))
        return 0

    # ---- A per-session step, on each session in turn -------------------
    # Which sessions "all" means depends on the step: those with something to adopt,
    # with the sort to freeze from, or with the curation to export.
    if args.step == "adopt":
        wanted = _sessions(cohort, args.sessions,
                           lambda r: (r.directory / args.old).is_dir())
    elif args.step == "freeze":
        wanted = _sessions(cohort, args.sessions, lambda r: args.sort in r.sorts)
    else:
        wanted = _sessions(cohort, args.sessions,
                           lambda r: args.curation in r.sorts.get(args.sort, {})
                           .get("curations", {}))

    failed = 0
    for number, record in enumerate(wanted, start=1):
        try:
            if args.step == "adopt":
                outcome = adopt_sort(record.directory, args.old, args.name)
            elif args.step == "freeze":
                outcome = freeze(record.sorts[args.sort]["dir"], args.curation,
                                 note=args.note)
            else:
                outcome = export(record, args.sort, args.curation)
            print(f"[{number}/{len(wanted)}] {record.session_id}: {outcome}", flush=True)
        except Exception as exc:                  # noqa: BLE001 — reported, not raised
            failed += 1
            print(f"[{number}/{len(wanted)}] {record.session_id}: FAILED — "
                  f"{type(exc).__name__}: {exc}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
