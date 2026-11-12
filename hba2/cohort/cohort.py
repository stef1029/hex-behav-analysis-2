"""
The cohort folder — where everything is.

A cohort is a directory of sessions, and this is the thing that knows what is in it.
Its whole job is answering *"where is the file I need"*: every path it hands out is
absolute and sits under a named key, so nothing downstream ever builds a path from a
session ID, globs a directory, or guesses a suffix.

``Cohort(path)`` walks the tree using directory listings only — no file is opened, so it
stays fast over a network filesystem — and attaches one NWB batch, the newest unless
``Cohort(path, batch=...)`` names another. Each session's record then says where its
batch file is, and which DeepLabCut model, sort and curation the batch uses for it.

``cohort.with_files(dlc=True, ephys=False)`` picks sessions by which kinds of data are
complete on disk; ``record.missing("ephys")`` says what one session lacks.

The page reads in the order the data moves::

    1. the file catalogue     what files a session can have, and their role names
    2. Cohort                 Cohort(path) → find_sessions → scan_session for each one,
                              then the batch; with_files to choose sessions
    3. finding sessions       find_sessions, is_session
    4. scan_session           one session folder → one SessionRecord, step by step
    5. SessionRecord          what comes out, and the questions you can ask it

Layout::

    <cohort>/                       base, passed in by the caller
    ├── nwb/<batch>/recipe.json     build_nwb's batches: what each was made from
    └── <group>/                    one per concurrent run across rigs
        ├── <session>/
        │   ├── raw/                the rig wrote this (or loose, in older sessions)
        │   ├── processed/          this package wrote this
        │   ├── dlc/<model>/        DeepLabCut wrote this
        │   ├── nwb/<batch>/        build_nwb wrote this: <session>.nwb in each batch
        │   └── ephys_sorts/<sort>/ a spike sort: the sorter's output, Phy, and the
        │       └── curations/<c>/  curations frozen from Phy — see ``ephys/``
        └── ephys/<Rig_mouse>/      Open Ephys wrote this, one folder per session

Only v2 control-system data is read, in **either of its two on-disk layouts**:

* ``nested`` — acquisition in ``<session>/raw``, output in ``<session>/processed``.
  What the rig writes now.
* ``flat`` — raw files loose in the session folder. What every v2 session recorded
  before the change looks like. Output still goes to ``<session>/processed``.

Which one a session uses is detected from the session itself, so nothing needs telling.

Only files this package produces and files the rig produces are catalogued. The previous
pipeline's artefacts — ``.nwb``, ``_analysis_v2.pkl``, ``_video_frame_times.json`` — are
deliberately invisible here; that pipeline is still the thing that reads them.

**No path here comes from the environment, a config file or a module constant.** The
base is passed in and everything derives from it, so moving a cohort changes one line at
the call site and nothing else.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# =====================================================================
# 1. The file catalogue
#
# Every file a session can have, under a role name. Rig files are matched on the end of
# the filename, never a glob: '_Tracker_data.json' must not also match the recovery
# marker '_Tracker_data.json_frame_id_recovery_processed', and '_video.avi' must not
# match DeepLabCut's '_videoDLC_<model>.csv'.
# =====================================================================

RAW_DIR = "raw"
PROCESSED_DIR = "processed"
DLC_DIR = "dlc"
EPHYS_DIR = "ephys"
SORTS_DIR = "ephys_sorts"
#: At cohort level, beside the groups: one folder per build_nwb batch, holding its
#: recipe. The files themselves are in each session's own folder of the same name.
BATCHES_DIR = "nwb"
NWB_DIR = "nwb"

#: Written by the rig during acquisition: role -> filename suffix. A role may list several
#: suffixes when the same file comes in more than one format; the first match in name
#: order wins.
RAW = {
    "metadata":         "-metadata.json",
    "trials_csv":       "-trials.csv",
    "blocks":           "-blocks.json",
    "daq":              "-ArduinoDAQ.h5",
    "scales_csv":       "_scales_data.csv",
    "tracker":          "_Tracker_data.json",
    "frame_ids_backup": "_frame_ids_backup.txt",
    # Older sessions have an .avi converted from the raw .bin after the session;
    # newer ones are encoded by the recorder as it runs, as .mkv (or .mp4).
    "video":            ("_video.avi", "_video.mkv", "_video.mp4"),
    # Which stillness model scored the session, and under what settings.
    # Written by the scales server beside its readings. Optional: a session
    # recorded before stillness existed, or with it switched off, has none.
    "scales_stillness": "_scales_stillness.json",
}

#: Written by this package's post-processing: role -> filename. These are **exact
#: filenames**, not suffixes: in the flat layout raw and processed share a folder, and a
#: suffix match on "trials.csv" would also catch the rig's "<session>-trials.csv".
PROCESSED = {
    "session_json": "session.json",
    "events":       "events.csv",
    "trials":       "trials.csv",
    # What the protocol annotated, on the PC clock — never the DAQ's. See
    # postprocessing/outputs/notes.py for why it is not columns on trials.csv.
    "notes":        "notes.csv",
    "frame_times":  "frame_times.csv",
    "scales":       "scales.csv",
    # Written by postprocessing/repairs/repair_video_timing.py, only for sessions whose video lost
    # frames to the old converter: what it decided, and post-processing's original.
    "video_repair":           "video_repair.json",
    "frame_times_unrepaired": "frame_times_unrepaired.csv",
    # Ephys sessions only: the ephys clock against the DAQ's, and how it was found.
    "ephys_sync":        "ephys_sync.csv",
    "ephys_sync_record": "ephys_sync.json",
}

#: Inside ``ephys_sorts/<sort>/``. The sorter's own output, the analyzer and Phy's folder are
#: kept whole under these names; ``sort.json`` says what produced them.
SORT = {
    "record":   "sort.json",
    "sorter":   "sorter",
    "analyzer": "analyzer",
    "phy":      "phy",
}
CURATIONS_DIR = "curations"

#: Inside ``ephys_sorts/<sort>/curations/<curation>/``: what Phy said when it was frozen, and
#: what this package derived from that. Exact filenames.
CURATION = {
    "record":         "curation.json",
    "cluster_group":  "cluster_group.tsv",
    "cluster_info":   "cluster_info.tsv",
    "spike_clusters": "spike_clusters.npy",
    "units":          "units.csv",
    "spikes":         "spikes.parquet",
    "waveforms":      "waveforms.npz",
    "names":          "neuron_names.csv",
}

#: Written by this package's pose stage, inside the model folder it was derived from.
#: Exact filenames, like PROCESSED and for the same reason — these share a directory
#: with DeepLabCut's own output and must never be mistaken for it.
POSE = {
    "pose":       "pose.csv",
    "trial_pose": "trial_pose.csv",
    "manifest":   "pose.json",
}

#: Written into ``dlc/<task>/`` by the analysis run. The folder is named for the Task,
#: which is the short name chosen in the model's config; this records the full model
#: name DeepLabCut generated around it, so the exact snapshot stays a recorded fact
#: rather than something parsed back out of a filename.
MODEL_RECORD = "model.json"

#: Marks a DeepLabCut output and separates the model name from the session prefix.
#: Used for sessions whose DLC output is still loose in the session folder; newer runs
#: put each model in ``<session>/dlc/<model>/`` and the folder name *is* the model.
DLC_MARKER = "_videoDLC_"

#: Pulse-train files are named after their channel, so the set is not known up front.
PULSE_PREFIX = "pulses_"

#: Roles without which post-processing cannot run.
REQUIRED = ("metadata", "trials_csv", "daq", "scales_csv", "tracker")

#: What post-processing writes for every session. With the REQUIRED rig files, these
#: make a session's behaviour complete. notes.csv is left out: sessions processed
#: before post-processing wrote it are complete without it.
BEHAVIOUR_FILES = ("session_json", "events", "trials", "frame_times", "scales")

#: The kinds of data SessionRecord.missing checks, and Cohort.with_files chooses by.
COMPLETE_KINDS = ("behaviour", "dlc", "pose", "ephys", "sort", "curation", "nwb")

#: Inside each batch folder: when it was made, what from, and what each session got.
RECIPE = "recipe.json"


# =====================================================================
# 2. The cohort
# =====================================================================

def model_for(model, record):
    """
    This session's DeepLabCut model, from a batch's ``model`` parameter: the one name
    given, or its mouse's entry, or the ``"*"`` default. None when no model is given.

    A mouse's entry may be a list in order of preference — the first this session has
    output for is used, so a session the preferred model never ran on falls back rather
    than dropping out. Raises KeyError when the mouse has no entry and there is no
    ``"*"`` default.
    """
    if model is None or isinstance(model, str):
        return model

    choice = model.get(record.mouse_id, model.get("*"))
    if choice is None:
        raise KeyError(f"no model for mouse {record.mouse_id} and no '*' default in {model}")
    if isinstance(choice, str):
        return choice

    for name in choice:
        key = record.dlc_key(name)
        # The pose stage reads the CSV; a model that left only its .h5 cannot be used.
        if key is not None and (record.dlc.get(key) or {}).get("csv") is not None:
            return name
    return choice[0]


def batch_cohort(nwb_path):
    """
    The cohort folder a batch file belongs to, from the file's path (Path or string):
    the folder above ``<session>/nwb/<batch>/<session>.nwb`` whose ``nwb/<batch>/``
    holds that batch's recipe. None for a file that is not in a batch.

    How a table of trials, which carries only each file's path, finds its cohort. The
    session may sit in a group folder or straight in the cohort, so the depth is not
    fixed.
    """
    nwb_path = Path(nwb_path)
    batch = nwb_path.parent.name
    for folder in nwb_path.parents:
        if (folder / BATCHES_DIR / batch / RECIPE).exists():
            return folder
    return None


class Cohort:
    """Every session in a cohort directory, where all their files are, and the NWB batch
    being read."""

    def __init__(self, base, batch=None):
        """
        Args:
            base: the cohort directory, as a Path or a string. The only absolute path in
                the system.
            batch: the NWB batch to read, by name. None means the newest — by the time
                its recipe says it was made — or no batch at all if build_nwb has not
                made one yet.
        """
        self.base = Path(base)
        self._asked_batch = batch
        self.load()

    def load(self):
        """
        Catalogue every session in the cohort folder, then attach the NWB batch. Call it
        again to pick up sessions or files that have appeared since.

        Directory listings only, apart from each batch's small recipe file and one
        look per session for its file in the batch being read.
        """
        # ---- 1. Every session folder, catalogued -------------------------
        #: session ID -> SessionRecord
        self._sessions = {}
        for directory in find_sessions(self.base):
            self._sessions[directory.name] = scan_session(directory)

        # ---- 2. Which batch ----------------------------------------------
        # Each batch's recipe.json says when it was made and what it was made from.
        # One that cannot be read (missing, or being written right now) counts as empty.
        recipes = {}
        for name, folder in self.batches.items():
            try:
                recipes[name] = json.loads((folder / RECIPE).read_text())
            except (OSError, ValueError):
                recipes[name] = {}

        if self._asked_batch is not None:
            if self._asked_batch not in recipes:
                raise ValueError(f"no batch {self._asked_batch!r} in {self.base.name}. "
                                 f"Batches: {', '.join(recipes) or 'none'}")
            self.batch = self._asked_batch
        elif recipes:
            # Newest by its recipe's time. One without a time counts as oldest, and
            # the name settles ties.
            self.batch = max(recipes, key=lambda name: (recipes[name].get("created_at", ""),
                                                        name))
        else:
            self.batch = None

        #: The batch's recipe.json: when it was made, its parameters, and what each
        #: session it built actually used. Empty when there is no batch.
        self.recipe = recipes.get(self.batch, {})

        # ---- 3. What the batch holds for each session --------------------
        # Its file, if written, and the model, sort and curation it uses for the session.
        # A session the batch has built records the model it actually got; sort and
        # curation come from the batch's parameters unless the session recorded its own
        # (a sessions.csv override), so a session missing its curation shows up as
        # missing it rather than as having none.
        parameters = self.recipe.get("parameters", {})
        built = self.recipe.get("sessions", {})

        for record in self._sessions.values():
            entry = built.get(record.session_id, {})

            if self.batch is not None:
                path = self.batch_path(self.batch, record.session_id)
                if path.exists():
                    record.nwb = path

            if entry.get("model"):
                record.batch_model = entry["model"]
            else:
                try:
                    record.batch_model = model_for(parameters.get("model"), record)
                except KeyError:
                    record.batch_model = None     # the batch names no model for this mouse

            record.batch_sort = entry.get("sort") or parameters.get("sort")
            record.batch_curation = entry.get("curation") or parameters.get("curation")

    # -- looking things up ---------------------------------------------

    def __len__(self):
        return len(self._sessions)

    def __iter__(self):
        return iter(self._sessions.values())

    def __contains__(self, session_id):
        return session_id in self._sessions

    @property
    def sessions(self):
        """Every session, keyed by ID."""
        return self._sessions

    @property
    def session_ids(self):
        return sorted(self._sessions)

    def session(self, session_id):
        """One session's record."""
        if session_id not in self._sessions:
            raise KeyError(
                f"No session {session_id!r} in {self.base.name}. "
                f"{len(self._sessions)} session(s) known.")
        return self._sessions[session_id]

    def with_files(self, *, processed=None, behaviour=None, dlc=None, pose=None,
                   ephys=None, sort=None, curation=None, nwb=None,
                   model=None, sort_name=None, curation_name=None):
        """
        Sessions sorted by ID, chosen by which kinds of data are complete on disk.

        For each kind — ``behaviour``, ``dlc``, ``pose``, ``ephys``, ``sort``,
        ``curation``, ``nwb`` — True keeps the sessions where it is complete, False keeps
        those where it is not (absent, or missing even one file), and leaving it out
        ignores it. :meth:`SessionRecord.missing` says what a session lacks.
        ``processed`` is True / False on whether post-processing has run at all.
        Nothing given means every session.

        The DeepLabCut model, sort and curation checked are the batch's, per session.
        ``model``, ``sort_name`` and ``curation_name`` check others instead — for a step
        that runs before any batch exists.

            cohort.with_files(dlc=True, pose=False)     # pose still to build
            cohort.with_files(ephys=True, sort=False)   # synced, not yet sorted
        """
        wanted = {"behaviour": behaviour, "dlc": dlc, "pose": pose, "ephys": ephys,
                  "sort": sort, "curation": curation, "nwb": nwb}

        chosen = []
        for session_id in self.session_ids:
            record = self._sessions[session_id]
            if processed is not None and record.is_processed != processed:
                continue

            keep = True
            for kind, must_be_complete in wanted.items():
                if must_be_complete is None:
                    continue
                is_complete = record.complete(kind, model=model, sort=sort_name,
                                              curation=curation_name)
                if is_complete != must_be_complete:
                    keep = False
                    break
            if keep:
                chosen.append(record)
        return chosen

    @property
    def mice(self):
        """Session IDs grouped by mouse."""
        grouped = {}
        for session_id in self.session_ids:
            mouse = self._sessions[session_id].mouse_id
            grouped.setdefault(mouse, []).append(session_id)
        return grouped

    @property
    def models(self):
        """
        Every DeepLabCut model in the cohort, and how many sessions carry it.

        How you find out what to ask for: the names here are what
        :meth:`SessionRecord.dlc_key` accepts, so nothing needs a config file to work
        out which coordinates exist.
        """
        counts = {}
        for record in self._sessions.values():
            for model in record.dlc:
                counts[model] = counts.get(model, 0) + 1
        return dict(sorted(counts.items()))

    @property
    def sorts(self):
        """Every spike sort name in the cohort, and how many sessions carry it."""
        counts = {}
        for record in self._sessions.values():
            for name in record.sorts:
                counts[name] = counts.get(name, 0) + 1
        return dict(sorted(counts.items()))

    @property
    def batches(self):
        """
        Every NWB batch build_nwb has made here: name -> the cohort-level folder
        holding its recipe.

        Read from the disk each time: a batch is often made after the cohort was
        scanned, and asking is rare enough that the listing is free.
        """
        root = self.base / BATCHES_DIR
        if not root.is_dir():
            return {}
        return {p.name: p for p in sorted(root.iterdir()) if p.is_dir()}

    def batch_path(self, batch, session_id):
        """
        Where a session's file in a batch is, whether or not it has been written:
        ``<session>/nwb/<batch>/<session>.nwb``.
        """
        return self.session(session_id).directory / NWB_DIR / batch / f"{session_id}.nwb"

    # -- reporting -----------------------------------------------------

    def describe(self):
        records = list(self._sessions.values())

        layouts = {}
        for record in records:
            layouts[record.layout] = layouts.get(record.layout, 0) + 1
        shape = ", ".join(f"{n} {name}" for name, n in sorted(layouts.items()))

        n_processed = sum(1 for r in records if r.is_processed)
        n_dlc = sum(1 for r in records if r.has_dlc)
        n_ephys = sum(1 for r in records if r.has_ephys)
        n_sorted = sum(1 for r in records if r.has_sort)

        lines = [
            f"{self.base.name}: {len(records)} session(s) ({shape})",
            f"  {len(self.mice)} mice · {n_processed} processed · {n_dlc} with DLC · "
            f"{n_ephys} with ephys · {n_sorted} sorted",
        ]

        if self.batch is None:
            lines.append("  no NWB batch yet")
        else:
            n_nwb = sum(1 for r in records if r.nwb is not None)
            lines.append(f"  batch {self.batch}: {n_nwb} NWB file(s)")

        duplicated = [r for r in records if r.dlc_duplicated]
        if duplicated:
            lines.append(
                f"  {len(duplicated)} session(s) have DLC output in both dlc/<model>/ "
                f"and loose in the session folder; the foldered copy is used")

        missing = [r for r in records if not r.can_process]
        if missing:
            lines.append(f"  {len(missing)} missing raw files:")
            for record in missing[:5]:
                lines.append(f"    {record.session_id}: {', '.join(record.missing_raw)}")
        return "\n".join(lines)


# =====================================================================
# 3. Finding the session folders
# =====================================================================

def find_sessions(base):
    """
    Every session directory under a cohort, given as a Path or a string.

    Cohorts nest two levels — ``<cohort>/<group>/<session>`` — because the rig groups
    concurrent sessions across rigs by start time. Non-session directories are skipped
    without being named: a group's ``ephys/`` and a session's ``processed/`` both fail
    the "holds a metadata file" test. The cohort's ``nwb/`` batches are skipped by name,
    since a batch folder can be large and is never a session.
    """
    found = []
    for group in sorted(p for p in Path(base).iterdir() if p.is_dir()):
        if group.name in (PROCESSED_DIR, DLC_DIR, EPHYS_DIR, BATCHES_DIR):
            continue
        # A session sitting directly in the cohort, with no group level.
        if is_session(group):
            found.append(group)
            continue
        for candidate in sorted(group.iterdir()):
            if is_session(candidate):
                found.append(candidate)
    return found


def is_session(directory):
    """True if ``directory`` (a Path or a string) is a session folder: it holds a
    ``*-metadata.json``, in ``raw/`` or loose."""
    directory = Path(directory)
    return directory.is_dir() and _metadata_folder(directory) is not None


def _metadata_folder(directory) -> Path | None:
    """Which folder holds the session's metadata file — ``raw/`` if the session is
    nested, the session folder itself if flat — or None if neither does."""
    suffix = RAW["metadata"]
    for candidate in (directory / RAW_DIR, directory):
        if not candidate.is_dir():
            continue
        for path in candidate.iterdir():
            if path.name.endswith(suffix) and path.is_file():
                return candidate
    return None


# =====================================================================
# 4. Reading one session folder
# =====================================================================

def scan_session(directory):
    """
    Catalogue one session, from its folder as a Path or a string. Raises if it is not
    a session folder.

    Directory listings only — nothing here opens a file, which matters on a network
    filesystem where a listing is cheap and an open is not.
    """
    directory = Path(directory)
    session_id = directory.name

    # ---- 1. Which layout ---------------------------------------------
    # The metadata file is in raw/ for a nested session, loose for a flat one.
    raw_dir = _metadata_folder(directory)
    if raw_dir is None:
        raise ValueError(
            f"{directory} is not a session folder: no '*-metadata.json' in "
            f"{directory}/{RAW_DIR} or {directory}"
        )
    if raw_dir.name == RAW_DIR:
        layout = "nested"
    else:
        layout = "flat"

    # ---- 2. Mouse and date, from the session ID ----------------------
    # The ID is <YYMMDD>_<HHMMSS>_<mouse>. The metadata file is authoritative for the
    # mouse; this is a fallback that needs no file opened.
    id_parts = session_id.split("_", 2)
    if len(id_parts) > 2:
        mouse_id = id_parts[2]
    else:
        mouse_id = ""
    if "_" in session_id:
        date = id_parts[0]
    else:
        date = ""

    # ---- 3. List the files -------------------------------------------
    # Output always lives in <session>/processed, whichever layout the raw files use.
    raw_files = [p for p in raw_dir.iterdir() if p.is_file()]
    processed_dir = directory / PROCESSED_DIR
    processed_files = []
    if processed_dir.is_dir():
        processed_files = [p for p in processed_dir.iterdir() if p.is_file()]

    # ---- 4. Rig files, matched on the end of the filename ------------
    # Every role appears, with None when absent, so the record shows what is missing.
    raw = {}
    for role, suffix in RAW.items():
        matches = sorted(p for p in raw_files if p.name.endswith(suffix))
        if matches:
            raw[role] = matches[0]
        else:
            raw[role] = None

    # ---- 5. Post-processing files, matched on the exact filename -----
    # Never on a suffix: in a flat session that would pick up the rig's
    # <session>-trials.csv as though it were the processed trials.csv.
    processed_by_name = {p.name: p for p in processed_files}
    processed = {}
    for role, filename in PROCESSED.items():
        processed[role] = processed_by_name.get(filename)

    # ---- 6. Pulse trains: pulses_<channel>.csv -----------------------
    pulses = {}
    for path in sorted(processed_files):
        if path.name.startswith(PULSE_PREFIX) and path.suffix == ".csv":
            channel = path.stem[len(PULSE_PREFIX):]
            pulses[channel] = path

    # ---- 7. DeepLabCut, and the pose stage's output ------------------
    # Two arrangements. Foldered, <session>/dlc/<model>/, is where new runs go and the
    # folder name is the model. Loose, ..._videoDLC_<model>.csv in the session folder,
    # is how DeepLabCut dumped it before. A file only counts as DeepLabCut output if its
    # name carries the marker — that keeps the pose stage's own pose.csv, which lives in
    # the model folder deliberately, from being mistaken for the coordinates.
    #
    # When a model is in both places the foldered copy wins: it got there deliberately.
    # The overlap is recorded rather than resolved silently, so a half-finished migration
    # is visible.
    #
    # A folder only counts for what it holds. DeepLabCut makes dlc/<model>/ before it
    # starts, so a job that fails leaves it empty, or with only its pickles; counting
    # that as a model would mark the video done and it would never be run again.
    dlc = {}
    pose = {}
    dlc_duplicated = set()

    dlc_dir = directory / DLC_DIR
    if dlc_dir.is_dir():
        for model_dir in sorted(p for p in dlc_dir.iterdir() if p.is_dir()):
            files_here = sorted(p for p in model_dir.iterdir() if p.is_file())
            found = {"csv": None, "h5": None}
            for path in files_here:
                kind = path.suffix.lower().lstrip(".")
                if DLC_MARKER in path.name and kind in found and found[kind] is None:
                    found[kind] = path
            if any(found.values()):
                dlc[model_dir.name] = found

            files_by_name = {p.name: p for p in files_here}
            roles = {role: files_by_name.get(filename) for role, filename in POSE.items()}
            if any(roles.values()):
                pose[model_dir.name] = roles

    # Loose files beside the video, in either listing; they only fill gaps.
    for path in raw_files + processed_files:
        if DLC_MARKER not in path.name:
            continue
        if path.suffix.lower() not in (".csv", ".h5"):
            continue

        after_marker = path.name.split(DLC_MARKER, 1)[1]
        model = after_marker.rsplit(".", 1)[0]
        kind = path.suffix.lower().lstrip(".")

        found = dlc.setdefault(model, {"csv": None, "h5": None})
        if found[kind] is None:
            found[kind] = path
        else:
            dlc_duplicated.add(model)

    # ---- 8. This mouse's ephys recording -----------------------------
    # One Open Ephys run covers every mouse started at the same time, in the group's
    # ephys/ folder, with a subfolder each named Rig<n>_<mouse>. Matched on the mouse
    # suffix, since rig numbering is not guaranteed to agree with the metadata. No match
    # means None, never a guess — ephys_root is kept for code that wants to look itself.
    ephys_root = directory.parent / EPHYS_DIR
    if not ephys_root.is_dir():
        ephys_root = None

    ephys = None
    if ephys_root is not None and mouse_id:
        for candidate in sorted(p for p in ephys_root.iterdir() if p.is_dir()):
            if candidate.name.endswith(f"_{mouse_id}"):
                ephys = candidate
                break

    # ---- 9. Spike sorts and their curations --------------------------
    # One folder per sort in <session>/ephys_sorts/, one per curation inside its
    # curations/. A folder counts as a sort whatever it holds — one half-moved or still
    # running is better listed with its gaps than not listed at all.
    sorts = {}
    sorts_dir = directory / SORTS_DIR
    if sorts_dir.is_dir():
        for sort_dir in sorted(p for p in sorts_dir.iterdir() if p.is_dir()):
            present = {p.name: p for p in sort_dir.iterdir()}

            curations = {}
            curations_dir = sort_dir / CURATIONS_DIR
            if curations_dir.is_dir():
                for curation_dir in sorted(p for p in curations_dir.iterdir() if p.is_dir()):
                    files_by_name = {p.name: p for p in curation_dir.iterdir() if p.is_file()}
                    curations[curation_dir.name] = {}
                    for role, filename in CURATION.items():
                        curations[curation_dir.name][role] = files_by_name.get(filename)

            sort = {"dir": sort_dir}
            for role, name in SORT.items():
                sort[role] = present.get(name)
            sort["curations"] = curations
            sorts[sort_dir.name] = sort

    # ---- 10. The record ----------------------------------------------
    return SessionRecord(
        session_id=session_id,
        mouse_id=mouse_id,
        date=date,
        directory=directory,
        layout=layout,
        raw=raw,
        processed=processed,
        pulses=pulses,
        dlc=dict(sorted(dlc.items())),
        dlc_duplicated=sorted(dlc_duplicated),
        pose=pose,
        ephys=ephys,
        ephys_root=ephys_root,
        sorts=sorts,
    )


# =====================================================================
# 5. One session's record
# =====================================================================

@dataclass
class SessionRecord:
    """
    Everything known about one session, as full paths.

    The contract: **if a file exists, this record holds its full path under a clearly
    named key.** Roles with no file are present as ``None`` rather than omitted, so the
    record also documents what a session *could* have.
    """

    session_id: str
    mouse_id: str
    date: str
    directory: Path

    #: ``nested`` (raw/ + processed/) or ``flat`` (everything loose).
    layout: str

    #: Acquisition files: role -> full path, or None.
    raw: dict = field(default_factory=dict)
    #: Derived files: role -> full path, or None. This package's own output only.
    processed: dict = field(default_factory=dict)
    #: Pulse trains: channel label -> full path. Only the ones that exist.
    pulses: dict = field(default_factory=dict)
    #: DeepLabCut output: model name -> {"csv": path, "h5": path}.
    dlc: dict = field(default_factory=dict)
    #: Models present both in ``dlc/<model>/`` and loose in the session folder. The
    #: foldered copy is the one used; this says the other is still there, so a
    #: half-finished migration is visible rather than silent.
    dlc_duplicated: list = field(default_factory=list)
    #: Pose-stage output: model name -> {"pose", "trial_pose", "manifest"}. Lives in
    #: the model's own folder, so it is keyed the same way the coordinates are.
    pose: dict = field(default_factory=dict)
    #: This session's own ephys recording, or None. One Open Ephys run covers every
    #: mouse started at the same time, but each gets its own subfolder, so this is the
    #: subfolder rather than the shared parent — nothing downstream should have to work
    #: out which one is theirs.
    ephys: Path | None = None
    #: The shared folder the above sits in, kept because the ephys layout changes and
    #: custom code may need to look around it.
    ephys_root: Path | None = None
    #: Spike sorts: sort name -> {"dir", "record", "sorter", "analyzer", "phy",
    #: "curations": {curation name -> {role -> path or None}}}.
    sorts: dict = field(default_factory=dict)

    # The cohort's NWB batch, as it applies to this session. Filled in by Cohort, not by
    # scan_session, and None when the cohort has no batch.

    #: This session's file in the batch, or None if the batch has not written one.
    nwb: Path | None = None
    #: The DeepLabCut model (Task name), spike sort and curation the batch uses for this
    #: session — what "complete" is checked against. None where it uses none.
    batch_model: str | None = None
    batch_sort: str | None = None
    batch_curation: str | None = None

    # -- processing state ----------------------------------------------

    @property
    def missing_raw(self):
        """Required acquisition files this session does not have."""
        missing = []
        for role in REQUIRED:
            if self.raw.get(role) is None:
                missing.append(role)
        return missing

    @property
    def can_process(self):
        """True if every file post-processing needs is here."""
        return len(self.missing_raw) == 0

    @property
    def is_processed(self):
        """True if this package has post-processed the session."""
        return self.processed.get("events") is not None

    @property
    def out_dir(self):
        """Where post-processing output goes — always ``<session>/processed``."""
        return self.directory / PROCESSED_DIR

    # -- DeepLabCut and pose -------------------------------------------

    @property
    def has_dlc(self):
        return len(self.dlc) > 0

    @property
    def dlc_models(self):
        """Model names with output for this session."""
        return sorted(self.dlc)

    def dlc_key(self, name) -> str | None:
        """
        This session's key for a model named loosely, or None if it has none.

        Models are asked for by **Task** — the short name from the config, which is
        what ``dlc/<task>/`` is called and what this returns unchanged. Sessions
        analysed before that convention are keyed by the longer name DeepLabCut built
        around the Task (``HrnetW32_<task>May27shuffle1_snapshot_best-190``), so a Task
        that appears inside exactly one of them resolves to it.

        Ambiguity raises rather than picking: two models sharing a Task substring are
        genuinely different answers, and silently choosing one is how a figure ends up
        built from a model nobody meant to use.
        """
        known = set(self.dlc) | set(self.pose)

        # A current session: the folder is named for the Task itself.
        if name in known:
            return name

        # An older session: look for the one long name with the Task inside it.
        matches = sorted(key for key in known if name in key)
        if len(matches) == 0:
            return None
        if len(matches) > 1:
            raise ValueError(
                f"{self.session_id}: '{name}' matches {len(matches)} models "
                f"({', '.join(matches)}) — name one exactly")
        return matches[0]

    def model_record(self, name) -> dict | None:
        """
        What the analysis run recorded about a model: full name, config, when it ran.

        None when the session has no such model, or has it from before the run started
        writing one.
        """
        key = self.dlc_key(name)
        if key is None:
            return None

        path = self.directory / DLC_DIR / key / MODEL_RECORD
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            return None

    @property
    def pose_models(self):
        """Model names this session has head angles for."""
        models = []
        for model, files in self.pose.items():
            if files.get("trial_pose"):
                models.append(model)
        return sorted(models)

    def has_pose(self, model=None):
        """Whether head angles exist — for one model, or for any."""
        if model:
            return model in self.pose_models
        return len(self.pose_models) > 0

    # -- ephys ---------------------------------------------------------

    @property
    def has_ephys(self):
        return self.ephys is not None

    @property
    def has_sort(self):
        return len(self.sorts) > 0

    def curation(self, sort, name):
        """
        One frozen curation's files, raising a useful error if either name is wrong.

        Every role in :data:`CURATION` is present, ``None`` where the file is not there
        yet — a curation that has been frozen but not exported has no ``units``.
        """
        if sort not in self.sorts:
            raise KeyError(f"{self.session_id} has no sort {sort!r}. "
                           f"Sorts: {', '.join(self.sorts) or 'none'}")

        curations = self.sorts[sort]["curations"]
        if name not in curations:
            raise KeyError(f"{self.session_id}: sort {sort!r} has no curation {name!r}. "
                           f"Curations: {', '.join(curations) or 'none'}")
        return curations[name]

    # -- completeness --------------------------------------------------

    def complete(self, kind, *, model=None, sort=None, curation=None):
        """True if nothing is missing for this kind of data — see :meth:`missing`."""
        return len(self.missing(kind, model=model, sort=sort, curation=curation)) == 0

    def missing(self, kind, *, model=None, sort=None, curation=None):
        """
        What this session lacks for one kind of data to be complete: empty when it is.

        One missing file is enough to make it incomplete. Why it is missing, and fixing
        it, is another script's job — this only says what is not there.

        Kinds (:data:`COMPLETE_KINDS`):

        ===========  ==========================================================
        behaviour    the REQUIRED rig files, and everything post-processing writes
        dlc          the model's DeepLabCut .csv and .h5
        pose         the model's pose.csv, trial_pose.csv and pose.json
        ephys        the Open Ephys recording, and its sync to the DAQ
        sort         the sort's sort.json, sorter/, analyzer/ and phy/
        curation     every file of the frozen curation
        nwb          this session's file in the cohort's batch
        ===========  ==========================================================

        The model, sort and curation default to what the cohort's batch uses for this
        session; pass them to check others.
        """
        if model is None:
            model = self.batch_model
        if sort is None:
            sort = self.batch_sort
        if curation is None:
            curation = self.batch_curation
        missing = []

        if kind == "behaviour":
            for role in self.missing_raw:
                missing.append(f"rig file {role}")
            for role in BEHAVIOUR_FILES:
                if self.processed.get(role) is None:
                    missing.append(PROCESSED[role])

        elif kind == "dlc":
            if model is None:
                return ["a DeepLabCut model to check: the batch names none"]
            key = self.dlc_key(model)
            if key is None or key not in self.dlc:
                return [f"DeepLabCut output for {model}"]
            for kind_of_file in ("csv", "h5"):
                if self.dlc[key].get(kind_of_file) is None:
                    missing.append(f"{model} .{kind_of_file}")

        elif kind == "pose":
            if model is None:
                return ["a DeepLabCut model to check: the batch names none"]
            key = self.dlc_key(model)
            files = self.pose.get(key, {})
            for role, filename in POSE.items():
                if files.get(role) is None:
                    missing.append(f"{model} {filename}")

        elif kind == "ephys":
            if self.ephys is None:
                missing.append("Open Ephys recording")
            for role in ("ephys_sync", "ephys_sync_record"):
                if self.processed.get(role) is None:
                    missing.append(PROCESSED[role])

        elif kind == "sort":
            if sort is None:
                return ["a sort to check: the batch names none"]
            if sort not in self.sorts:
                return [f"sort {sort}"]
            for role, name in SORT.items():
                if self.sorts[sort].get(role) is None:
                    missing.append(f"{sort}/{name}")

        elif kind == "curation":
            if sort is None or curation is None:
                return ["a sort and curation to check: the batch names none"]
            curations = self.sorts.get(sort, {}).get("curations", {})
            if curation not in curations:
                return [f"curation {sort}/{curation}"]
            for role, filename in CURATION.items():
                if curations[curation].get(role) is None:
                    missing.append(f"{sort}/{curation}/{filename}")

        elif kind == "nwb":
            if self.nwb is None:
                missing.append("NWB file in the batch")

        else:
            raise ValueError(f"unknown kind {kind!r}. Kinds: {', '.join(COMPLETE_KINDS)}")

        return missing

    # -- access --------------------------------------------------------

    def path(self, role):
        """
        Full path for ``role``, raising a useful error rather than returning None.

        Use when the file is required; read ``.raw``/``.processed`` directly when its
        absence is a normal state you want to branch on.
        """
        for files in (self.raw, self.processed):
            if role not in files:
                continue
            if files[role] is None:
                raise FileNotFoundError(f"{self.session_id} has no {role!r} file")
            return files[role]

        known = sorted(set(self.raw) | set(self.processed))
        raise KeyError(f"Unknown role {role!r}. Known roles: {', '.join(known)}")

    def describe(self):
        """One line: the layout, and how many of each kind of file the session has."""
        raw_found = sum(1 for path in self.raw.values() if path is not None)
        processed_found = sum(1 for path in self.processed.values() if path is not None)

        parts = [
            f"{self.session_id}  [{self.layout}]",
            f"raw {raw_found}/{len(self.raw)}",
            f"processed {processed_found}/{len(self.processed)}",
            f"dlc {len(self.dlc)}",
            f"pulses {len(self.pulses)}",
        ]
        if self.has_ephys:
            parts.append("ephys")
        for name, sort in self.sorts.items():
            parts.append(f"sort {name} ({len(sort['curations'])} curation(s))")
        return "  ".join(parts)
