"""Final agent — strongest Minesweeper solver in the project.

Inherits the full CSP + probability pipeline from ProbabilityAgent and adds:
1. Smarter tie-breaking when multiple cells share the minimum mine probability.
2. Endgame exact solving when few unrevealed cells remain.
"""
from __future__ import annotations

import math

import numpy as np

from minesweeper.board import CellState

from .csp_agent import Constraint, NEIGHBORS
from .probability_agent import ProbabilityAgent

Cell = tuple[int, int]


class FinalAgent(ProbabilityAgent):
    name = "final"

    def reset(self) -> None:
        super().reset()
        self._last_view: np.ndarray | None = None

    def act(self, view: np.ndarray):
        self._last_view = view
        self.last_was_guess = False
        self.last_probabilities = {}
        return super(ProbabilityAgent, self).act(view)

    def _fallback(self, view: np.ndarray):
        unrevealed = self.unrevealed_cells(view)
        if not unrevealed:
            return ("reveal", 0, 0)

        flagged_count = int((view == int(CellState.FLAGGED)).sum())
        remaining_mines = self.n_mines - flagged_count

        if len(unrevealed) <= 25:
            result = self._endgame_solve(view, unrevealed, remaining_mines)
            if result is not None:
                return result

        return super()._fallback(view)


    def _endgame_solve(self, view: np.ndarray,
                       unrevealed: list[Cell],
                       remaining_mines: int):
        constraints = self._build_constraints(view)
        unrevealed_set = set(unrevealed)
        ordered = sorted(unrevealed_set)
        n = len(ordered)
        cell_to_idx = {cell: i for i, cell in enumerate(ordered)}

        con_info: list[tuple[frozenset[int], int]] = []
        for con in constraints:
            valid_cells = con.cells & unrevealed_set
            if not valid_cells:
                continue
            indices = frozenset(cell_to_idx[c] for c in valid_cells)
            con_info.append((indices, con.n_mines))

        cell_to_cons: list[list[int]] = [[] for _ in range(n)]
        for ci, (indices, _) in enumerate(con_info):
            for idx in indices:
                cell_to_cons[idx].append(ci)

        UNDECIDED = -1
        assignment = [UNDECIDED] * n
        solutions: list[list[int]] = []
        max_solutions = 100_000

        def feasible(pos: int) -> bool:
            for ci in cell_to_cons[pos]:
                indices, required = con_info[ci]
                mines = 0
                undecided = 0
                for idx in indices:
                    v = assignment[idx]
                    if v == 1:
                        mines += 1
                    elif v == UNDECIDED:
                        undecided += 1
                if mines > required or mines + undecided < required:
                    return False
            return True

        def dfs(i: int, mines_so_far: int) -> bool:
            if mines_so_far > remaining_mines:
                return True
            if i == n:
                if mines_so_far == remaining_mines:
                    solutions.append(list(assignment))
                return len(solutions) < max_solutions
            remaining_cells = n - i
            if mines_so_far + remaining_cells < remaining_mines:
                return True
            for value in (0, 1):
                assignment[i] = value
                new_mines = mines_so_far + value
                if new_mines <= remaining_mines and feasible(i):
                    if not dfs(i + 1, new_mines):
                        assignment[i] = UNDECIDED
                        return False
            assignment[i] = UNDECIDED
            return True

        completed = dfs(0, 0)
        if not completed or not solutions:
            return None

        mine_counts = [0] * n
        for sol in solutions:
            for j in range(n):
                mine_counts[j] += sol[j]

        total = len(solutions)
        probabilities: dict[Cell, float] = {}
        for j, cell in enumerate(ordered):
            probabilities[cell] = mine_counts[j] / total

        certain_safe = [cell for cell, p in probabilities.items() if p == 0.0]
        if certain_safe:
            r, c = certain_safe[0]
            self.last_reason = "endgame: certain safe"
            return ("reveal", r, c)

        certain_mine = [cell for cell, p in probabilities.items() if p == 1.0]
        if certain_mine:
            r, c = certain_mine[0]
            self.last_reason = "endgame: certain mine"
            return ("flag", r, c)

        self.last_was_guess = True
        self.last_probabilities = probabilities
        frontier = set()
        for con in constraints:
            frontier.update(con.cells & set(ordered))
        action = self._select_cell(probabilities, frontier)
        _, r, c = action
        p = probabilities[(r, c)]
        self.last_reason = f"endgame: P(mine)={p:.1%}"
        return action

    def _select_cell(self, probabilities, frontier_cells):
        min_p = min(probabilities.values())
        eps = 1e-9
        candidates = [c for c, p in probabilities.items() if abs(p - min_p) < eps]

        if len(candidates) == 1:
            r, c = candidates[0]
            return ("reveal", r, c)

        view = self._last_view
        UNREVEALED = int(CellState.UNREVEALED)

        scored: list[tuple[float, int, int]] = []
        for (r, c) in candidates:
            score = 0.0
            if (r, c) in frontier_cells:
                score += 10.0
            unrevealed_neighbors = 0
            for dr, dc in NEIGHBORS:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < self.h and 0 <= nc < self.w):
                    continue
                if view is not None:
                    nv = int(view[nr, nc])
                    if 0 <= nv <= 8:
                        score += 1.0
                    elif nv == UNREVEALED:
                        unrevealed_neighbors += 1
            score += unrevealed_neighbors * 3.0
            max_dist = math.hypot(self.h / 2, self.w / 2)
            dist = math.hypot(r - self.h / 2, c - self.w / 2)
            score += (1 - dist / max_dist) * 2.0 if max_dist > 0 else 0.0
            scored.append((score, r, c))

        scored.sort(key=lambda x: -x[0])
        best_score = scored[0][0]
        best = [(r, c) for s, r, c in scored if abs(s - best_score) < eps]
        idx = int(self.rng.integers(len(best)))
        r, c = best[idx]
        return ("reveal", r, c)
