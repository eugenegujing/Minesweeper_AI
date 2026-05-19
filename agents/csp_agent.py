"""Constraint-satisfaction agent.

Each revealed numeric cell with value n imposes the constraint
    sum(unrevealed neighbors that are mines) == n - flagged_neighbors
on its unrevealed neighbors. We apply two trivial rules and a subset
reasoning step. When no certain action is found, fall back to a random
unrevealed cell (the hybrid agent replaces this fallback with a probability
estimator).

Subset rule:
  If constraint A's variable set is a subset of B's, then B - A is a
  derived constraint with sum (n_B - n_A). If this derived sum equals 0,
  all those cells are safe; if it equals their cardinality, all are mines.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from minesweeper.board import CellState

from .base import Action, Agent


NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1),
             (0, -1),           (0, 1),
             (1, -1),  (1, 0),  (1, 1)]


@dataclass(frozen=True)
class Constraint:
    cells: frozenset[tuple[int, int]]
    n_mines: int


class CSPAgent(Agent):
    name = "csp"

    def reset(self) -> None:
        super().reset()
        self._pending_safe: list[tuple[int, int]] = []
        self._pending_flag: list[tuple[int, int]] = []
        self._first_move = True

    def act(self, view: np.ndarray) -> Action:
        self.last_was_guess = False
        if self._first_move:
            self._first_move = False
            self.last_reason = "first move (center)"
            return ("reveal", self.h // 2, self.w // 2)

        if not self._pending_safe and not self._pending_flag:
            self._infer(view)

        while self._pending_flag:
            r, c = self._pending_flag.pop()
            if view[r, c] == int(CellState.UNREVEALED):
                self.last_reason = "CSP-confirmed mine"
                return ("flag", r, c)
        while self._pending_safe:
            r, c = self._pending_safe.pop()
            if view[r, c] == int(CellState.UNREVEALED):
                self.last_reason = "CSP-confirmed safe"
                return ("reveal", r, c)

        # No certain move; fall back to a random unrevealed cell.
        return self._fallback(view)

    # ------------------------------------------------------------------
    def _fallback(self, view: np.ndarray) -> Action:
        cells = self.unrevealed_cells(view)
        if not cells:
            return ("reveal", 0, 0)
        self.last_was_guess = True
        self.last_reason = "random fallback (CSP stuck)"
        idx = int(self.rng.integers(len(cells)))
        r, c = cells[idx]
        return ("reveal", r, c)

    # ------------------------------------------------------------------
    def _build_constraints(self, view: np.ndarray) -> list[Constraint]:
        out: list[Constraint] = []
        for r in range(self.h):
            for c in range(self.w):
                v = int(view[r, c])
                if not (0 <= v <= 8):
                    continue
                unrevealed: list[tuple[int, int]] = []
                flagged = 0
                for dr, dc in NEIGHBORS:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < self.h and 0 <= nc < self.w):
                        continue
                    nv = int(view[nr, nc])
                    if nv == int(CellState.UNREVEALED):
                        unrevealed.append((nr, nc))
                    elif nv == int(CellState.FLAGGED):
                        flagged += 1
                if unrevealed:
                    out.append(Constraint(frozenset(unrevealed), v - flagged))
        return out

    def _infer(self, view: np.ndarray) -> None:
        constraints = self._build_constraints(view)
        safe: set[tuple[int, int]] = set()
        mines: set[tuple[int, int]] = set()

        # Iterate until no new info found (subset reasoning is monotonic).
        changed = True
        guard = 0
        while changed and guard < 50:
            changed = False
            guard += 1
            new_constraints: list[Constraint] = []
            for con in constraints:
                # Trivial rules
                if con.n_mines == 0:
                    for cell in con.cells:
                        if cell not in safe:
                            safe.add(cell); changed = True
                    continue
                if con.n_mines == len(con.cells):
                    for cell in con.cells:
                        if cell not in mines:
                            mines.add(cell); changed = True
                    continue
                new_constraints.append(con)
            # Apply known cells to constraints (remove resolved variables).
            reduced: list[Constraint] = []
            for con in new_constraints:
                cells = con.cells - safe - mines
                n = con.n_mines - sum(1 for cell in con.cells if cell in mines)
                if not cells:
                    continue
                if (cells, n) != (con.cells, con.n_mines):
                    changed = True
                reduced.append(Constraint(frozenset(cells), n))
            # Subset reasoning: if A.cells subset of B.cells, derive B-A.
            derived: list[Constraint] = []
            for a in reduced:
                for b in reduced:
                    if a is b: continue
                    if a.cells < b.cells:  # strict subset
                        diff = b.cells - a.cells
                        n = b.n_mines - a.n_mines
                        if not diff:
                            continue
                        derived.append(Constraint(diff, n))
            # Deduplicate
            constraints = list({(c.cells, c.n_mines): c
                                for c in (reduced + derived)}.values())
            if derived:
                changed = True

        self._pending_safe = list(safe)
        self._pending_flag = list(mines)
