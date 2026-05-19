"""Single-point baseline agent.

Applies only the two single-point Minesweeper rules. No subset reasoning,
no probability. Falls back to a random unrevealed cell when neither rule
fires. This is the baseline that CSPAgent and the probability / final
agents must beat.
"""
from __future__ import annotations

import numpy as np

from minesweeper.board import CellState

from .base import Action, Agent


NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1),
             (0, -1),           (0, 1),
             (1, -1),  (1, 0),  (1, 1)]


class SinglePointAgent(Agent):
    name = "single_point"

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
                self.last_reason = "single-point rule 2: must be a mine"
                return ("flag", r, c)

        while self._pending_safe:
            r, c = self._pending_safe.pop()
            if view[r, c] == int(CellState.UNREVEALED):
                self.last_reason = "single-point rule 1: must be safe"
                return ("reveal", r, c)

        return self._fallback(view)

    def _infer(self, view: np.ndarray) -> None:
        safe: set[tuple[int, int]] = set()
        mines: set[tuple[int, int]] = set()

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

                if not unrevealed:
                    continue

                remaining = v - flagged

                if remaining == 0:
                    safe.update(unrevealed)
                elif remaining == len(unrevealed):
                    mines.update(unrevealed)

        self._pending_safe = list(safe)
        self._pending_flag = list(mines)

    def _fallback(self, view: np.ndarray) -> Action:
        cells = self.unrevealed_cells(view)
        if not cells:
            return ("reveal", 0, 0)
        self.last_was_guess = True
        self.last_reason = "random fallback (single-point stuck)"
        idx = int(self.rng.integers(len(cells)))
        r, c = cells[idx]
        return ("reveal", r, c)
