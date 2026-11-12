"""
Reader for ``<session>-blocks.json``.

Which blocks the session ran, and **the parameters they ran under** — cue duration,
response timeout, which ports were enabled. The trials CSV carries only the block
*name*, so without this the experimental conditions of every trial are unrecorded, and
answering "how did accuracy depend on cue duration?" means looking the block up in the
control system source, which changes over time.

Written by the rig from the block engine's own record, so it is what actually ran rather
than what a definition says today.

Absent on every session recorded before the rig wrote it. That is not an error: the
reader returns an empty record and post-processing carries on, because a missing
optional file is a normal state and a pipeline that stops for one is a pipeline people
learn to work around.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


def read(path):
    """
    Parse a block record (a Path, a string, or None) into a :class:`Blocks`. ``None``
    or a missing file gives an empty record.

    A malformed file is treated the same way rather than raising: it is optional
    metadata, and losing it should not cost a session its post-processing.
    """
    if path is None:
        return Blocks(path=None)

    path = Path(path)
    if not path.exists():
        return Blocks(path=None)

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return Blocks(path=None)

    entered = []
    for entry in document.get("blocks", []):
        # Unix seconds as written by the engine; None if absent or unreadable.
        try:
            entered_at = datetime.fromtimestamp(float(entry.get("entered_at")))
        except (TypeError, ValueError, OSError):
            entered_at = None
        entered.append(Block(
            name=entry.get("name", ""),
            display_name=entry.get("display_name", ""),
            is_warmup=bool(entry.get("is_warmup", False)),
            entered_at=entered_at,
            params=entry.get("params") or {},
        ))

    return Blocks(path=path, blocks=entered)


@dataclass(frozen=True)
class Block:
    """One block, with the parameters it resolved to when it ran."""

    name: str
    display_name: str
    is_warmup: bool
    entered_at: datetime | None
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Blocks:
    """Every block a session entered, in order."""

    path: Path | None
    blocks: list[Block] = field(default_factory=list)

    def __len__(self):
        return len(self.blocks)

    def __iter__(self):
        return iter(self.blocks)

    @property
    def recorded(self):
        """False when the session predates the rig writing this file."""
        return self.path is not None

    @property
    def names(self):
        """Distinct block names, in first-seen order."""
        seen = {}
        for block in self.blocks:
            seen.setdefault(block.name, None)
        return list(seen)

    def describe(self):
        if not self.recorded:
            return "blocks: not recorded (session predates the block record)"
        return f"blocks: {len(self.blocks)} entered, {len(self.names)} distinct"
