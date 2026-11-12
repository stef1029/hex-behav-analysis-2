"""
Which videos still need DeepLabCut, and where their output goes.
================================================================

Writes one tab-separated line per video that has no output yet for a given model::

    <video path>	<destination folder>	<session id>

Three fields rather than one, because the array job needs all three and only the cohort
knows them. The destination is ``<session>/dlc/<task>/``, so DeepLabCut writes into the
model's own folder instead of beside the video — which in a nested session would mean
writing into ``raw/``. The session ID is carried rather than re-derived from the path,
which only works for one of the two layouts.

The model is always identified by the **Task** name in its config, so running an
analysis means naming a config file and nothing else.

Run directly, the way the submission script calls it::

    python make_vid_list.py <cohort folder> <config.yaml> [output file]
"""

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from hba2 import Cohort  # noqa: E402

DLC_DIR = "dlc"


def model_task(config_path):
    """
    The ``Task`` name from a DeepLabCut config (a path, Path or string) — the model's
    identity everywhere here.

    DeepLabCut embeds it in every file it writes, and it is the one part of a model's
    name that is chosen rather than generated, so it is what names the output folder and
    what you quote later to ask for these coordinates.
    """
    config = yaml.safe_load(Path(config_path).read_text())
    task = None
    if isinstance(config, dict):
        task = config.get("Task")
    if not task:
        raise ValueError(f"no 'Task' field in {config_path}")
    return str(task)


def destination(record, task):
    """Where this model's coordinates for this session belong."""
    return record.directory / DLC_DIR / task


def has_model(record, task):
    """
    Whether this session already has output from this model.

    The matching rule lives in :meth:`SessionRecord.dlc_key` — a folder named exactly
    for the Task, or the longer generated name a session analysed before that
    convention is keyed by.

    An ambiguous name still means the work was done, so it does not re-run: two models
    sharing a Task substring is a problem for whoever reads the coordinates, not for
    whoever decides whether to produce them, and it is raised there instead of
    aborting a whole submission here.
    """
    try:
        return record.dlc_key(task) is not None
    except ValueError:
        return True


def videos_needing_model(cohort, task):
    """Every session with a compressed video and no output yet for this model."""
    return [record for record in cohort
            if record.raw.get("video") and not has_model(record, task)]


def write_list(records, task, output):
    """Write the three-column list the array job indexes into, to ``output`` (a Path)."""
    lines = [f"{record.raw['video']}\t{destination(record, task)}\t{record.session_id}\n"
             for record in records]
    output.write_text("".join(lines))
    return output


def main():
    argv = sys.argv[1:]
    if len(argv) < 2:
        print(__doc__)
        return 1

    directory, config = Path(argv[0]), argv[1]
    if not directory.is_dir():
        print(f"Error: cohort directory does not exist: {directory}")
        return 1

    try:
        task = model_task(config)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"Error: could not read a model name from {config}: {exc}")
        return 1

    cohort = Cohort(directory)
    with_video = [r for r in cohort if r.raw.get("video")]
    needed = videos_needing_model(cohort, task)
    output = Path(argv[2]) if len(argv) > 2 else directory / f"dlc_videos_{task}.tsv"

    print(f"Cohort: {directory}")
    print(f"Model:  {task}  (from {config})")
    print(f"\n{len(cohort)} session(s), {len(with_video)} with a compressed video")
    print(f"  needing analysis: {len(needed)}")
    print(f"  already done    : {len(with_video) - len(needed)}")
    if len(cohort) != len(with_video):
        print(f"  no video yet    : {len(cohort) - len(with_video)} "
              "(still waiting on compression)")

    write_list(needed, task, output)
    print(f"\nWrote {output}")
    for record in needed:
        print(f"  {record.session_id} -> {destination(record, task)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
