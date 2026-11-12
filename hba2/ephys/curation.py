"""
Freezing a curation: what Phy says now, copied somewhere it cannot change.

Phy's state is three files it rewrites on every save — the labels, its summary of each
cluster, and which cluster every spike is in (which merges and splits change). Copied
together under a name, they are a curation that later Phy work cannot disturb, and that
a figure can cite.

Phy on Windows writes those files with ``\\r\\n`` line endings, which read as a label of
``"good\\r"`` that matches nothing. They are fixed on the way in, so nothing downstream
has to know.

Nothing about a curation is derived here. Its units, spikes and waveforms come from
:func:`~.export.export`, which reads only the frozen copy — so an export can always be
redone from the curation alone, whatever Phy has done since.
"""

import getpass
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..cohort import CURATION, CURATIONS_DIR, SORT

#: Phy's files, as named in its folder, and the role each has in a curation.
FROM_PHY = {"cluster_group": "cluster_group.tsv", "cluster_info": "cluster_info.tsv",
            "spike_clusters": "spike_clusters.npy"}


def freeze(sort_dir, name, *, note=""):
    """
    Copy Phy's current labels, cluster summary and spike assignment into a curation.

    Args:
        sort_dir: ``<session>/ephys_sorts/<sort>``, as a Path or a string.
        name: what to call it — a date and who did it reads well later
            (``2026-09_manual``). ``auto`` is the conventional name for the sorter's own
            labels, frozen before anyone opened Phy.
        note: anything worth knowing later, stored in ``curation.json``.

    Refuses to overwrite an existing curation of that name: freeze under a new one.
    Returns the curation's folder, a Path.
    """
    sort_dir = Path(sort_dir)
    phy = sort_dir / SORT["phy"]
    target = sort_dir / CURATIONS_DIR / name
    if not (phy / FROM_PHY["cluster_group"]).exists():
        raise FileNotFoundError(f"no cluster_group.tsv in {phy}")
    if target.exists():
        raise FileExistsError(f"curation {name!r} already exists at {target}")

    # ---- 1. Copy Phy's files, fixing line endings ------------------------
    target.mkdir(parents=True)
    fixed = []
    for role, filename in FROM_PHY.items():
        source = phy / filename
        if not source.exists():
            continue
        if source.suffix == ".tsv":
            raw = source.read_bytes()
            if b"\r" in raw:
                fixed.append(filename)
            (target / CURATION[role]).write_bytes(raw.replace(b"\r", b""))
        else:
            shutil.copy2(source, target / CURATION[role])

    # ---- 2. What the copy holds ------------------------------------------
    labels = pd.read_csv(target / CURATION["cluster_group"], sep="\t")
    labels["group"] = labels["group"].astype(str).str.strip()
    spike_clusters = np.load(target / CURATION["spike_clusters"], mmap_mode="r")
    in_spikes = set(np.unique(spike_clusters).tolist())
    labelled = set(labels["cluster_id"].tolist())
    templates = np.load(phy / "spike_templates.npy", mmap_mode="r")
    if len(templates):
        n_templates = int(templates.max()) + 1
    else:
        n_templates = 0

    # When Phy last saved each file: the curation is Phy's state as of then.
    #: Phy's filename -> when it was last saved, as ISO text
    saved = {}
    for filename in FROM_PHY.values():
        if (phy / filename).exists():
            saved[filename] = (datetime.fromtimestamp((phy / filename).stat().st_mtime)
                               .isoformat(timespec="seconds"))

    # ---- 3. curation.json ------------------------------------------------
    record = {
        "name": name,
        "frozen_at": datetime.now().isoformat(timespec="seconds"),
        "frozen_by": getpass.getuser(),
        "note": note,
        "phy": {
            "folder": str(phy),
            "saved": saved,
            "line_endings_fixed": fixed,
        },
        "labels": {group: int(n) for group, n in labels["group"].value_counts().items()},
        "n_clusters": len(in_spikes),
        # Cluster IDs past the sorter's templates were made in Phy, by a merge or split.
        "n_made_in_phy": sum(1 for c in in_spikes if c >= n_templates),
        # Phy treats a cluster with no label as unsorted; it is never exported.
        "unlabelled_clusters": sorted(c for c in in_spikes if c not in labelled)[:50],
    }
    (target / CURATION["record"]).write_text(json.dumps(record, indent=2),
                                             encoding="utf-8")
    return target
