"""
Moving a sort from the pipeline's old folder into ``ephys_sorts/<name>/``.

The sorting pipeline wrote one folder per sorter beside the behaviour::

    <session>/spikesorter_output_<x>/        <session>/ephys_sorts/<name>/
    ├── 0/ … 7/  logs/                 ->    ├── sorter/0/ … 7/  logs/
    ├── pipeline_provenance.json       ->    │   └── pipeline_provenance.json
    └── combined/                            ├── analyzer/   (was sorting_analyzer_combined)
        ├── sorting_analyzer_combined  ->    ├── phy/        (was phy_combined)
        └── phy_combined               ->    └── sort.json   written here

**Moves, not copies**: on one filesystem a move is a rename, so a 100 GB Phy folder
costs nothing. Phy's ``params.py`` holds the absolute path of its ``recording.dat``, which
is rewritten to the new place so Phy still opens.

Nothing is curated here. Freezing a curation is :func:`~.curation.freeze`, afterwards.
"""

import json
import re
import shutil
from datetime import datetime
from pathlib import Path

from ..cohort import CURATIONS_DIR, SORT, SORTS_DIR

#: The pipeline's folder, and what this sort is called once moved.
DEFAULT_OLD = "spikesorter_output_kilosort4_camnt"
DEFAULT_NAME = "ks4_camnt"

_COMBINED = "combined"
_OLD_ANALYZER = "sorting_analyzer_combined"
_OLD_PHY = "phy_combined"
_PROVENANCE = "pipeline_provenance.json"


def adopt_sort(session_dir, old=DEFAULT_OLD, name=DEFAULT_NAME):
    """
    Move ``<session>/<old>/`` into ``<session>/ephys_sorts/<name>/`` and record what it is.

    ``session_dir`` is a Path or a string. Refuses if the destination exists: two sorts
    under one name is the confusion this layout exists to prevent. Returns the new sort
    folder, a Path.
    """
    session_dir = Path(session_dir)
    source = session_dir / old
    sort_dir = session_dir / SORTS_DIR / name
    if not source.is_dir():
        raise FileNotFoundError(f"no {old}/ in {session_dir}")
    if sort_dir.exists():
        raise FileExistsError(f"{sort_dir} already exists")

    # ---- 1. What the pipeline said it did --------------------------------
    # Read before anything moves. None if it is missing or not JSON.
    try:
        provenance = json.loads((source / _PROVENANCE).read_text())
    except (OSError, ValueError):
        provenance = None
    old_phy = source / _COMBINED / _OLD_PHY

    # ---- 2. The new folder -----------------------------------------------
    sorter = sort_dir / SORT["sorter"]
    sorter.mkdir(parents=True)
    (sort_dir / CURATIONS_DIR).mkdir()

    # ---- 3. Phy's folder and the analyzer, to their own names ------------
    moved = {}
    if old_phy.is_dir():
        shutil.move(str(old_phy), sort_dir / SORT["phy"])
        moved["phy"] = str(old_phy)
    analyzer = source / _COMBINED / _OLD_ANALYZER
    if analyzer.is_dir():
        shutil.move(str(analyzer), sort_dir / SORT["analyzer"])
        moved["analyzer"] = str(analyzer)

    # ---- 4. Everything else, under sorter/ -------------------------------
    # Shank folders, logs, provenance, anything a later version adds — moved whole
    # rather than being judged here.
    if (source / _COMBINED).is_dir():
        for leftover in sorted((source / _COMBINED).iterdir()):
            shutil.move(str(leftover), sorter / f"combined_{leftover.name}")
    for entry in sorted(source.iterdir()):
        if entry.name != _COMBINED:
            shutil.move(str(entry), sorter / entry.name)
    for empty in (source / _COMBINED, source):
        if empty.is_dir() and not any(empty.iterdir()):
            empty.rmdir()

    # ---- 5. Point Phy at the recording's new place -----------------------
    # Only Phy's dat_path line is rewritten; the rest of params.py is left alone.
    phy = sort_dir / SORT["phy"]
    params = phy / "params.py"
    if params.exists():
        recording = phy / "recording.dat"
        text = params.read_text()
        updated = re.sub(r"^dat_path\s*=.*$", f"dat_path = r'{recording}'", text,
                         flags=re.MULTILINE)
        if updated != text:
            params.write_text(updated)

    # ---- 6. sort.json ----------------------------------------------------
    record = {
        "name": name,
        "adopted_from": str(source),
        "adopted_at": datetime.now().isoformat(timespec="seconds"),
        "moved": moved,
        # What the sorting pipeline said it did, kept whole: preprocessing, sorter
        # parameters, which raw stream and recording, and its own QC counts.
        "pipeline": provenance,
        "probe": {
            "geometry": "phy/channel_positions.npy, phy/channel_groups.npy",
            # The headstage contact-to-channel map was a best guess when these sorts
            # ran; shank and depth inherit whatever it got right.
            "wiring_verified": False,
        },
    }
    (sort_dir / SORT["record"]).write_text(json.dumps(record, indent=2), encoding="utf-8")
    return sort_dir
