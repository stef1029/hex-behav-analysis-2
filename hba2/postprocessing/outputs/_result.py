"""The shape every output module returns."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class OutputResult:
    """What one output module wrote, and anything it noticed while writing."""

    name: str
    path: Path | None
    n_rows: int = 0
    n_unmatched: int = 0
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def written(self):
        return self.path is not None

    def describe(self):
        if not self.written:
            return f"{self.name:<14} not written"
        line = f"{self.name:<14} {self.n_rows:>9,} rows"
        if self.n_unmatched:
            line += f" ({self.n_unmatched:,} unmatched)"
        return line
