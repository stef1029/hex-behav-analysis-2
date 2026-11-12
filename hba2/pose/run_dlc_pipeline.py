#!/usr/bin/env python3
"""
Submit DeepLabCut analysis for a cohort.
========================================

One pipeline, four steps: pick the videos that still need analysing, divide them into
groups, write a list per group, submit an array job per list. How the work is grouped
is an argument, so a new way of dividing it is a parameter rather than another script.

    submit_dlc(cohort, config)                        one model, one job
    submit_dlc(cohort, {"M1": cfg_a, "M2": cfg_b}) a model per mouse
    submit_dlc(cohort, config, users={...})            split across users' quota
    submit_dlc(cohort, {...}, users={...})             both at once

**A model is named by the Task field in its config**, so running an analysis means
giving a config path and nothing else. That Task also names the folder the coordinates
land in — ``<session>/dlc/<task>/`` — so output never goes near the raw acquisition
files, and asking for it later means quoting the same short name you submitted with.

**Everything a run produces lands in the cohort folder**: the video list and one log
per session, named for that session. So the state of a run is visible in one place and
clears out in one command.

Usage:
    On the cluster, with pose and NWB once DeepLabCut is done (this is what the
    control app's post-processing submits, through slurm/cohort_pipeline.sh):
        python run_dlc_pipeline.py <cohort> --mouse-model T1=<config.yaml> \\
            --scripts-dir <slurm/> [--env <venv or conda env>] [--gpus 8] \\
            [--then-nwb] [--dry-run]

    From any machine, from Python — over ssh, and split across accounts if you like:
        from hba2.pose.run_dlc_pipeline import submit_dlc
        submit_dlc("/path/to/cohort", "/path/to/config.yaml", num_gpus=4,
                   env="/path/to/envs/hba2", ssh_host="cluster-login", ssh_user="me")

**The environment every job runs in** is ``env``: a virtual environment's folder or a
conda environment's name, as the cluster sees it. The job scripts pass it on to the
jobs they submit. Left as None, each job uses ``HBA2_ENV`` from the shell that
submitted it.

    Keep the call with your own cohorts, models and accounts in a script of your own,
    outside this package, so the package holds no lab's values.

The page, top to bottom::

    submit_dlc          the pipeline, from any machine
    plan_groups         what to run: which videos, under which model, by whom
    sbatch_command,
    _run_via_ssh,
    _run_local          running it
    cli, run_cohort     on the cluster: DeepLabCut, then pose and NWB once it is done
    Group               one array job's worth of work
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hba2 import Cohort, SessionRecord  # noqa: E402

#: Everything this pipeline drives lives beside it in ``slurm/``. Added to the path
#: explicitly rather than relying on relative imports, so the file behaves the same run
#: as a script and imported as a module.
SCRIPTS = Path(__file__).resolve().parent / "slurm"
sys.path.insert(0, str(SCRIPTS))
from make_vid_list import destination, has_model, model_task, write_list  # noqa: E402

SUBMISSION_SCRIPT = "unified_DLC_analysis_submission.sh"


# =============================================================================
# The pipeline
# =============================================================================

def submit_dlc(cohort, model_config, *, num_gpus=8, users=None, scripts_dir=None,
               env=None, ssh_host=None, ssh_user=None, ssh_port=22, ssh_key_path=None,
               dry_run=False):
    """
    Submit DeepLabCut analysis for everything in a cohort that still needs it.

    Args:
        cohort: a :class:`Cohort`, or a path (Path or string) to build one from.
        model_config: a path to a DLC ``config.yaml``, or ``{mouse: config path}`` to
            give different mice different models in one submission.
        num_gpus: concurrent array slots per job.
        users: ``{username: ssh settings}`` to spread the work over several accounts'
            cluster quota; each gets an equal share and submits as itself. ``None``
            submits everything as you. The settings are a dict that may hold
            ``ssh_host``, ``ssh_user``, ``ssh_port`` and ``ssh_key_path``.
        scripts_dir: where the pipeline's own scripts are, a Path or a string. Defaults
            to ``slurm/`` beside this file, which is only right when the cluster sees
            this checkout at the same path as the machine you call from.
        env: the environment the jobs run in — a venv folder or a conda env name, as
            the cluster sees it. None leaves it to ``HBA2_ENV`` on the cluster.
        ssh_host: submit through this host instead of locally. Ignored when ``users``
            is given, since each user carries its own connection settings.
        ssh_user, ssh_port, ssh_key_path: how to connect to ``ssh_host``.
        dry_run: print what would be submitted and stop.

    Returns:
        0 when every group submitted cleanly, otherwise the first non-zero code.
    """
    if scripts_dir:
        scripts_dir = Path(scripts_dir)
    else:
        scripts_dir = SCRIPTS
    if not isinstance(cohort, Cohort):
        cohort = Cohort(cohort)
    cohort_folder = cohort.base

    # ---- 1. What to run -------------------------------------------------
    user_names = None
    if users:
        user_names = list(users)
    groups = plan_groups(cohort, model_config, user_names)

    print("=" * 66)
    print("  DeepLabCut submission")
    print("=" * 66)
    print(f"  Cohort      : {cohort_folder}  ({len(cohort)} sessions)")
    print(f"  GPUs/job    : {num_gpus}")
    print(f"  Scripts     : {scripts_dir}")
    print(f"  Environment : {env or '$HBA2_ENV on the cluster'}")
    if not groups:
        print("\n  Nothing to do — every session already has output for this model.")
        print("=" * 66)
        return 0

    for group in groups:
        who = f" as {group.user}" if group.user else ""
        print(f"  {group.task}{who}: {len(group.records)} video(s)")
    print("=" * 66)

    # ---- 2. A list and a submission per group ------------------------------
    worst = 0
    for group in groups:
        list_path = cohort_folder / group.list_path_name
        if dry_run:
            print(f"\n[DRY RUN] Would write {list_path} "
                  f"({len(group.records)} videos)")
            for record in group.records[:3]:
                print(f"    {record.session_id} -> "
                      f"{destination(record, group.task)}")
            if len(group.records) > 3:
                print(f"    ... and {len(group.records) - 3} more")
        else:
            write_list(group.records, group.task, list_path)
            print(f"\nWrote {list_path} ({len(group.records)} videos)")

        command = sbatch_command(scripts_dir, cohort_folder, group,
                                 list_path, num_gpus, env=env)

        # Spread over several users, each user's own settings say where to connect;
        # otherwise the ssh_* arguments do.
        settings = (users or {}).get(group.user) or {}
        if users:
            default_host = None
        else:
            default_host = ssh_host
        host = settings.get("ssh_host", default_host)
        if host:
            code = _run_via_ssh(
                command, host=host,
                user=settings.get("ssh_user", group.user or ssh_user or ""),
                port=settings.get("ssh_port", ssh_port),
                key_path=settings.get("ssh_key_path", ssh_key_path),
                dry_run=dry_run)
        else:
            code = _run_local(command, dry_run=dry_run)

        worst = worst or code

    return worst


# =============================================================================
# Deciding what to run
# =============================================================================

def resolve_configs(model_config):
    """
    Turn the model argument into ``{config path: Task}``, reading each config once.

    A single path (a string) applies to every mouse; a mapping ``{mouse: config path}``
    assigns a model per mouse, which is how one cohort gets analysed with several models
    in one submission.
    """
    if isinstance(model_config, str):
        paths = {model_config}
    else:
        paths = set(model_config.values())
    return {path: model_task(path) for path in sorted(paths)}


def split_evenly(items, parts):
    """Divide a list into ``parts`` contiguous chunks (a list of lists) of near-equal size."""
    if parts < 1:
        raise ValueError("cannot split into fewer than one part")
    size, extra = divmod(len(items), parts)
    chunks, start = [], 0
    for index in range(parts):
        # The first ``extra`` chunks take one more each.
        stop = start + size
        if index < extra:
            stop += 1
        chunks.append(items[start:stop])
        start = stop
    return chunks


def plan_groups(cohort, model_config, users=None):
    """
    Work out what to submit, as a list of :class:`Group`: which videos, under which
    model, by whom.

    ``model_config`` is as :func:`resolve_configs` takes it; ``users`` a list of user
    names to share the work between, or None.

    A session is included when it has a compressed video and no output yet for that
    model — the two conditions that make up "still needs analysing". Groups with no
    work are dropped rather than submitted empty.
    """
    tasks = resolve_configs(model_config)
    per_mouse = isinstance(model_config, dict)

    groups = []
    for config, task in tasks.items():
        # A model per mouse: just the mice given this config. Otherwise every mouse.
        mice = None
        if per_mouse:
            mice = {mouse for mouse, cfg in model_config.items() if cfg == config}

        records = [record for record in cohort
                   if (mice is None or record.mouse_id in mice)
                   and record.raw.get("video")
                   and not has_model(record, task)]
        if not records:
            continue

        if users:
            for user, chunk in zip(users, split_evenly(records, len(users))):
                if chunk:
                    groups.append(Group(task, config, chunk, user))
        else:
            groups.append(Group(task, config, records))

    return groups


# =============================================================================
# Running things
# =============================================================================

def sbatch_command(scripts_dir, cohort_folder, group, list_path, num_gpus, env=None):
    """
    The submission for one group, as one shell-quoted string.

    ``--cohort`` is passed alongside ``--videolist`` deliberately: the list is already
    built, and naming the cohort is what keeps the per-session logs there.
    """
    parts = [
        "sbatch", str(scripts_dir / SUBMISSION_SCRIPT),
        "--videolist", str(list_path),
        "--cohort", str(cohort_folder),
        "--config", str(group.config),
        "--gpus", str(num_gpus),
        "--scripts-dir", str(scripts_dir),
    ]
    if env:
        parts += ["--env", str(env)]
    return " ".join(shlex.quote(part) for part in parts)


def _dry_run_prefix(dry_run):
    """What each printed command starts with: a marker on a dry run, nothing otherwise."""
    if dry_run:
        return "[DRY RUN] "
    return ""


def _run_via_ssh(command, host, user, port=22, key_path=None, dry_run=False):
    """
    Run a command on a remote machine, and return its exit code.

    Uses the ``ssh`` binary rather than a library so agent forwarding, ``~/.ssh/config``
    host aliases and ProxyJump chains all work without being reimplemented here.
    """
    ssh_parts = ["ssh", "-p", str(port)]
    if key_path:
        ssh_parts += ["-i", str(Path(key_path).expanduser())]
    ssh_parts += [f"{user}@{host}", command]

    printable = " ".join(shlex.quote(p) for p in ssh_parts)
    print(f"\n{_dry_run_prefix(dry_run)}Running via SSH on {user}@{host}:")
    print(f"  $ {printable}\n")
    if dry_run:
        return 0
    return subprocess.run(ssh_parts, text=True,
                          stdout=sys.stdout, stderr=sys.stderr).returncode


def _run_local(command, dry_run=False):
    """Run a shell command on this machine, and return its exit code."""
    print(f"\n{_dry_run_prefix(dry_run)}Running locally:")
    print(f"  $ {command}\n")
    if dry_run:
        return 0
    return subprocess.run(command, shell=True, text=True,
                          stdout=sys.stdout, stderr=sys.stderr).returncode


# =============================================================================
# On the cluster: DeepLabCut, then pose and NWB once it has finished
# =============================================================================
#
# submit_dlc() goes through unified_DLC_analysis_submission.sh, which submits the GPU
# array from inside its own job — so the array's job ID never comes back, and nothing
# can be made to wait for it. When this runs on the cluster itself there is no need for
# that hop: the arrays are submitted directly, their IDs collected, and one after-DLC
# job waits for all of them. One job rather than one per model, because each would
# rewrite the cohort's sessions.csv.

ARRAY_SCRIPT = "submit_analysis_DLC3.sh"
AFTER_DLC_SCRIPT = "after_dlc.sh"


def parse_mouse_models(pairs):
    """``["T1=/path/config.yaml", ...]`` to ``{"T1": "/path/config.yaml"}``."""
    models = {}
    for pair in pairs:
        mouse, sep, config = pair.partition("=")
        if not sep or not mouse or not config:
            raise ValueError(f"expected MOUSE=CONFIG, got {pair!r}")
        if mouse in models and models[mouse] != config:
            raise ValueError(f"mouse {mouse!r} given two models")
        models[mouse] = config
    return models


def cli(argv=None):
    """
    The command line used by cohort_pipeline.sh, and by hand on the cluster.

    ``argv`` is a list of arguments, or None for ``sys.argv``.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Submit DeepLabCut (and optionally pose + NWB) for a cohort, "
                    "from the cluster.")
    parser.add_argument("cohort", help="cohort folder, as the cluster sees it")
    parser.add_argument("--mouse-model", action="append", required=True,
                        metavar="MOUSE=CONFIG",
                        help="a mouse and its DeepLabCut config.yaml; repeat per mouse")
    parser.add_argument("--gpus", type=int, default=8,
                        help="concurrent array slots per model")
    parser.add_argument("--scripts-dir", required=True,
                        help="this pipeline's slurm/ folder, as the cluster sees it")
    parser.add_argument("--env", default=None,
                        help="venv folder or conda env name every job runs in "
                             "(default: $HBA2_ENV)")
    parser.add_argument("--then-nwb", action="store_true",
                        help="afterwards, build pose and NWB for what succeeded")
    parser.add_argument("--dry-run", action="store_true",
                        help="print what would be submitted and stop")
    args = parser.parse_args(argv)

    return run_cohort(args.cohort, parse_mouse_models(args.mouse_model),
                      num_gpus=args.gpus, scripts_dir=args.scripts_dir, env=args.env,
                      then_nwb=args.then_nwb, dry_run=args.dry_run)


def run_cohort(cohort, mouse_models, *, num_gpus=8, scripts_dir=None, env=None,
               then_nwb=False, dry_run=False):
    """
    Everything a cohort still needs, submitted from the cluster.

    ``cohort`` is a :class:`Cohort` or a path to one; ``mouse_models`` is
    ``{mouse: config.yaml path}``; ``scripts_dir`` a Path or a string, defaulting to
    ``slurm/`` beside this file; ``env`` the environment the jobs run in, a venv folder
    or a conda env name — None hands on ``HBA2_ENV`` from this process, which is how
    cohort_pipeline.sh passes its own ``--env`` along.

    DeepLabCut for every session of a listed mouse without output from its model;
    then, with ``then_nwb``, one job for pose and NWB. That job is submitted even when
    there is no DeepLabCut to run, so sessions analysed earlier but never built still
    are. Every stage skips what is already current, so running this again only does
    what is new.
    """
    if scripts_dir:
        scripts_dir = Path(scripts_dir)
    else:
        scripts_dir = SCRIPTS
    if not isinstance(cohort, Cohort):
        cohort = Cohort(cohort)

    print("=" * 66)
    print("  Cluster analysis")
    print("=" * 66)
    print(f"  Cohort   : {cohort.base}  ({len(cohort)} sessions)")
    env = env or os.environ.get("HBA2_ENV")
    print(f"  Scripts  : {scripts_dir}")
    print(f"  Env      : {env or '<none: the jobs will stop, with no --env or HBA2_ENV>'}")
    for mouse, config in sorted(mouse_models.items()):
        print(f"  {mouse:<8} : {config}")

    # Nothing silently dropped: a mouse with sessions here and no model is named.
    unmodelled = sorted({r.mouse_id for r in cohort} - set(mouse_models))
    if unmodelled:
        print(f"  No model, not analysed: {', '.join(unmodelled)}")
    print("=" * 66)

    job_ids = submit_arrays(cohort, mouse_models, num_gpus=num_gpus,
                            scripts_dir=scripts_dir, env=env, dry_run=dry_run)
    if not job_ids:
        print("\nNo DeepLabCut to run — every listed mouse's sessions have output.")

    if then_nwb:
        submit_after_dlc(cohort.base, mouse_models, scripts_dir=scripts_dir,
                         after=job_ids, env=env, dry_run=dry_run)
    return 0


def _sbatch(parts, env=None, dry_run=False):
    """
    Submit, and return the job ID ``--parsable`` printed.

    ``parts`` is the command as a list of strings; ``env`` a dict of extra environment
    variables, or None.
    """
    command_line = " ".join(shlex.quote(p) for p in parts)
    print(f"\n{_dry_run_prefix(dry_run)}$ {command_line}")
    if dry_run:
        return "DRYRUN"
    done = subprocess.run(parts, text=True, capture_output=True,
                          env={**os.environ, **(env or {})})
    if done.returncode != 0:
        raise RuntimeError(f"sbatch exited with code {done.returncode}: "
                           f"{done.stderr.strip()}")
    job_id = done.stdout.strip().split(";")[0]
    print(f"  submitted job {job_id}")
    return job_id


def submit_arrays(cohort, mouse_models, *, num_gpus, scripts_dir, env=None,
                  dry_run=False):
    """
    Submit a DeepLabCut array per model, from the cluster, and return their job IDs.

    Same plan, list files and per-session logs as :func:`submit_dlc`; only the
    submission differs, so that the IDs come back.
    """
    job_ids = []
    for group in plan_groups(cohort, mouse_models):
        list_path = cohort.base / group.list_path_name
        print(f"\n{group.task}: {len(group.records)} video(s)")
        if not dry_run:
            write_list(group.records, group.task, list_path)
            try:   # group-writable for other lab members' runs; only works if ours
                list_path.chmod(list_path.stat().st_mode | 0o060)
            except OSError:
                pass
        job_ids.append(_sbatch(
            ["sbatch", f"--array=1-{len(group.records)}%{num_gpus}",
             f"--chdir={cohort.base}", "--parsable",
             str(scripts_dir / ARRAY_SCRIPT)],
            # What unified_DLC_analysis_submission.sh would have exported. The list is
            # named relative to --chdir, as it is there.
            env={"DLC_VIDEO_LIST_FILE": list_path.name,
                 "DLC_MODEL_CONFIG": group.config,
                 "DLC_SCRIPTS_DIR": str(scripts_dir),
                 **_env_var(env)},
            dry_run=dry_run))
    return job_ids


def submit_after_dlc(cohort_folder, mouse_models, *, scripts_dir, after, env=None,
                     dry_run=False):
    """
    Submit pose, the session index and NWB for the cohort, waiting for ``after`` (a
    list of job IDs), and return its own job ID.

    ``afterany`` rather than ``afterok``: one failed video must not hold back every
    other session's NWB. Sessions without DeepLabCut output are skipped, with a reason.
    """
    parts = ["sbatch", "--parsable", f"--chdir={cohort_folder}",
             f"--output={cohort_folder / 'after_dlc_%j.log'}"]
    if after:
        parts.append("--dependency=afterany:" + ":".join(after))
    parts += [str(scripts_dir / AFTER_DLC_SCRIPT),
              "--scripts-dir", str(scripts_dir), str(cohort_folder)]
    for mouse, config in sorted(mouse_models.items()):
        parts += ["--mouse-model", f"{mouse}={config}"]
    if env:
        parts += ["--env", str(env)]
    return _sbatch(parts, dry_run=dry_run)


def _env_var(env):
    """``{"HBA2_ENV": env}`` for a job's environment, or nothing when there is none."""
    if env:
        return {"HBA2_ENV": str(env)}
    return {}


# =============================================================================
# What one submission is
# =============================================================================

@dataclass
class Group:
    """One array job's worth of work: a model, and the videos to run it on."""

    task: str
    config: str
    records: list[SessionRecord] = field(default_factory=list)
    #: Which account submits this one, when the work is spread over several.
    user: str | None = None

    @property
    def name(self):
        return f"{self.task}_{self.user}" if self.user else self.task

    @property
    def list_path_name(self):
        return f"dlc_videos_{self.name}.tsv"


if __name__ == "__main__":
    sys.exit(cli())
