"""Core Minesweeper board logic, independent of any RL framework."""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable, Optional

import numpy as np


class CellState(IntEnum):
    """Public view of a cell. Numeric counts 0..8 are stored as 0..8 directly;
    these constants only cover non-numeric states."""
    UNREVEALED = -1
    FLAGGED = -2
    MINE = -3   # only revealed after a loss


DIFFICULTIES: dict[str, tuple[int, int, int]] = {
    "beginner":     (9, 9, 10),
    "intermediate": (16, 16, 40),
    "expert":       (16, 30, 99),
}


@dataclass
class StepResult:
    view: np.ndarray
    reward: float
    terminated: bool
    won: bool
    info: dict


class Board:
    """A Minesweeper board.

    Mines are placed lazily on the first reveal so the first click is always safe
    (and ideally opens a zero-region, matching standard Minesweeper rules).
    """

    def __init__(self, height: int, width: int, n_mines: int,
                 rng: Optional[np.random.Generator] = None,
                 safe_first_click: bool = True):
        if n_mines >= height * width:
            raise ValueError("n_mines must be < height * width")
        self.h = height
        self.w = width
        self.n_mines = n_mines
        self.rng = rng if rng is not None else np.random.default_rng()
        self.safe_first_click = safe_first_click

        self._mines = np.zeros((height, width), dtype=bool)
        self._revealed = np.zeros((height, width), dtype=bool)
        self._flagged = np.zeros((height, width), dtype=bool)
        self._counts = np.zeros((height, width), dtype=np.int8)
        self._mines_placed = False
        self.first_action: Optional[tuple[int, int]] = None
        self.lost = False
        self.won = False

    # ------------------------------------------------------------------
    # Mine placement
    # ------------------------------------------------------------------
    def _place_mines(self, safe_r: int, safe_c: int) -> None:
        forbidden: set[tuple[int, int]] = set()
        if self.safe_first_click:
            for r, c in self._neighbors(safe_r, safe_c):
                forbidden.add((r, c))
            forbidden.add((safe_r, safe_c))
        all_cells = [(r, c) for r in range(self.h) for c in range(self.w)
                     if (r, c) not in forbidden]
        # If too many forbidden cells (tiny boards), fall back to just the click cell.
        if len(all_cells) < self.n_mines:
            all_cells = [(r, c) for r in range(self.h) for c in range(self.w)
                         if (r, c) != (safe_r, safe_c)]
        idx = self.rng.choice(len(all_cells), size=self.n_mines, replace=False)
        for k in idx:
            r, c = all_cells[k]
            self._mines[r, c] = True
        # Precompute neighbor counts.
        for r in range(self.h):
            for c in range(self.w):
                if self._mines[r, c]:
                    self._counts[r, c] = -1
                else:
                    self._counts[r, c] = sum(
                        self._mines[nr, nc] for nr, nc in self._neighbors(r, c)
                    )
        self._mines_placed = True

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    def _neighbors(self, r: int, c: int) -> Iterable[tuple[int, int]]:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.h and 0 <= nc < self.w:
                    yield nr, nc

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.h and 0 <= c < self.w

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def reveal(self, r: int, c: int) -> bool:
        """Reveal a cell. Returns True if the cell was safe, False if it was a mine.

        Flood-fills zero-regions automatically.
        """
        if self.lost or self.won:
            return not self.lost
        if not self.in_bounds(r, c):
            raise ValueError(f"({r},{c}) out of bounds")
        if self._flagged[r, c] or self._revealed[r, c]:
            return True
        if not self._mines_placed:
            self.first_action = (r, c)
            self._place_mines(r, c)
        if self._mines[r, c]:
            self.lost = True
            self._revealed[r, c] = True
            return False
        self._flood_reveal(r, c)
        self._check_win()
        return True

    def _flood_reveal(self, r: int, c: int) -> None:
        stack = [(r, c)]
        while stack:
            cr, cc = stack.pop()
            if self._revealed[cr, cc] or self._flagged[cr, cc]:
                continue
            self._revealed[cr, cc] = True
            if self._counts[cr, cc] == 0:
                for nr, nc in self._neighbors(cr, cc):
                    if not self._revealed[nr, nc] and not self._mines[nr, nc]:
                        stack.append((nr, nc))

    def toggle_flag(self, r: int, c: int) -> None:
        if self.lost or self.won:
            return
        if self._revealed[r, c]:
            return
        self._flagged[r, c] = not self._flagged[r, c]

    def _check_win(self) -> None:
        # Win when all non-mine cells are revealed.
        if (self._revealed | self._mines).all():
            self.won = True

    # ------------------------------------------------------------------
    # Public observation
    # ------------------------------------------------------------------
    def view(self) -> np.ndarray:
        """Return an int array the agent can observe. Revealed cells contain 0..8;
        unrevealed contain CellState.UNREVEALED, flagged contain CellState.FLAGGED,
        and revealed mines (after a loss) contain CellState.MINE."""
        out = np.full((self.h, self.w), int(CellState.UNREVEALED), dtype=np.int8)
        out[self._flagged] = int(CellState.FLAGGED)
        revealed = self._revealed
        out[revealed] = self._counts[revealed]
        if self.lost:
            out[self._mines & self._revealed] = int(CellState.MINE)
        return out

    @property
    def n_remaining(self) -> int:
        """Cells still unrevealed (excluding flagged)."""
        return int((~self._revealed & ~self._flagged).sum())

    @property
    def n_flags(self) -> int:
        return int(self._flagged.sum())

    def render_ascii(self) -> str:
        symbols = {
            int(CellState.UNREVEALED): ".",
            int(CellState.FLAGGED): "F",
            int(CellState.MINE): "*",
        }
        v = self.view()
        rows = []
        for r in range(self.h):
            row = []
            for c in range(self.w):
                val = int(v[r, c])
                if val in symbols:
                    row.append(symbols[val])
                elif val == 0:
                    row.append(" ")
                else:
                    row.append(str(val))
            rows.append(" ".join(row))
        return "\n".join(rows)
