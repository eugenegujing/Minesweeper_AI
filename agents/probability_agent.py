"""Probability-based agent.

Computes mine probability for each unrevealed cell when CSP cannot find a
certain safe/mine cell. Picks the cell with minimum mine probability,
preferring frontier cells on ties (revealing them yields more information).

Algorithm (overrides CSP's random _fallback):
1. Build constraints from the current view (reuse CSPAgent._build_constraints).
2. Partition unrevealed cells into frontier (in some constraint) and
   off-frontier (in no constraint).
3. Split the frontier into connected components.
4. Enumerate every valid mine assignment for each component (DFS).
5. Weight assignments by C(off-frontier-size, remaining-mines - frontier-mines).
6. Aggregate marginal P(mine) per cell, including off-frontier.
7. Pick min P(mine). Tiebreaker: prefer frontier.
"""
from __future__ import annotations

import math

import numpy as np

from minesweeper.board import CellState

from .csp_agent import CSPAgent, Constraint


Cell = tuple[int, int]
Component = tuple[frozenset[Cell], list[Constraint]]


class ProbabilityAgent(CSPAgent):
    name = "probability"

    def reset(self) -> None:
        super().reset()
        self.last_probabilities: dict[Cell, float] = {}

    def act(self, view: np.ndarray):
        self.last_probabilities = {}
        return super().act(view)

    def _find_frontier(
        self, view: np.ndarray, constraints: list[Constraint]
    ) -> tuple[set[Cell], set[Cell]]:
        frontier: set[Cell] = set()
        for con in constraints:
            frontier.update(con.cells)

        off_frontier: set[Cell] = set()
        unrevealed = int(CellState.UNREVEALED)
        for r in range(self.h):
            for c in range(self.w):
                if int(view[r, c]) == unrevealed and (r, c) not in frontier:
                    off_frontier.add((r, c))

        return frontier, off_frontier

    def _split_components(
        self,
        frontier_cells: set[Cell],
        constraints: list[Constraint],
    ) -> list[Component]:
        cell_to_constraints: dict[Cell, list[Constraint]] = {}
        for con in constraints:
            for cell in con.cells:
                cell_to_constraints.setdefault(cell, []).append(con)

        visited: set[Cell] = set()
        components: list[Component] = []

        for seed in frontier_cells:
            if seed in visited:
                continue
            comp_cells: set[Cell] = set()
            comp_constraints: list[Constraint] = []
            seen_constraints: set[Constraint] = set()
            stack = [seed]
            while stack:
                cell = stack.pop()
                if cell in visited:
                    continue
                visited.add(cell)
                comp_cells.add(cell)
                for con in cell_to_constraints.get(cell, []):
                    if con in seen_constraints:
                        continue
                    seen_constraints.add(con)
                    comp_constraints.append(con)
                    for neighbor in con.cells:
                        if neighbor not in visited:
                            stack.append(neighbor)
            components.append((frozenset(comp_cells), comp_constraints))

        return components

    def _enumerate_component(
        self, component: Component
    ) -> list[tuple[dict[Cell, int], int]]:
        cells, constraints = component
        ordered_cells: list[Cell] = sorted(cells)
        n = len(ordered_cells)
        cell_to_idx = {cell: i for i, cell in enumerate(ordered_cells)}

        con_info: list[tuple[frozenset[int], int]] = [
            (frozenset(cell_to_idx[c] for c in con.cells), con.n_mines)
            for con in constraints
        ]

        cell_to_cons: list[list[int]] = [[] for _ in range(n)]
        for ci, (indices, _) in enumerate(con_info):
            for idx in indices:
                cell_to_cons[idx].append(ci)

        UNDECIDED = -1
        assignment = [UNDECIDED] * n
        solutions: list[tuple[dict[Cell, int], int]] = []

        def feasible(touched: list[int]) -> bool:
            for ci in touched:
                indices, required = con_info[ci]
                decided_mines = 0
                undecided = 0
                for idx in indices:
                    v = assignment[idx]
                    if v == 1:
                        decided_mines += 1
                    elif v == UNDECIDED:
                        undecided += 1
                if decided_mines > required:
                    return False
                if decided_mines + undecided < required:
                    return False
            return True

        def dfs(i: int) -> None:
            if i == n:
                mine_count = sum(1 for v in assignment if v == 1)
                snapshot = {ordered_cells[j]: assignment[j] for j in range(n)}
                solutions.append((snapshot, mine_count))
                return
            for value in (0, 1):
                assignment[i] = value
                if feasible(cell_to_cons[i]):
                    dfs(i + 1)
            assignment[i] = UNDECIDED

        dfs(0)
        return solutions

    def _compute_probabilities(
        self,
        components: list[Component],
        off_frontier_cells: set[Cell],
        view: np.ndarray,
    ) -> dict[Cell, float]:
        F = len(off_frontier_cells)
        flagged_count = int((view == int(CellState.FLAGGED)).sum())
        M = self.n_mines - flagged_count

        h_totals: list[dict[int, int]] = []
        h_cells: list[dict[Cell, dict[int, int]]] = []

        for comp in components:
            comp_cells, _ = comp
            sols = self._enumerate_component(comp)
            if not sols:
                return {}

            h_total: dict[int, int] = {}
            h_cell: dict[Cell, dict[int, int]] = {cell: {} for cell in comp_cells}
            for config, k in sols:
                h_total[k] = h_total.get(k, 0) + 1
                for cell, value in config.items():
                    if value == 1:
                        h_cell[cell][k] = h_cell[cell].get(k, 0) + 1

            h_totals.append(h_total)
            h_cells.append(h_cell)

        n = len(components)
        prefix: list[dict[int, int]] = [{0: 1}]
        for h in h_totals:
            prefix.append(self._convolve(prefix[-1], h))

        suffix: list[dict[int, int]] = [{0: 1}] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix[i] = self._convolve(suffix[i + 1], h_totals[i])

        H_global = prefix[n]

        def weighted_sum(hist: dict[int, int]) -> float:
            total = 0.0
            for k, count in hist.items():
                leftover = M - k
                if 0 <= leftover <= F:
                    total += count * math.comb(F, leftover)
            return total

        total_weight = weighted_sum(H_global)
        if total_weight == 0:
            return {}

        probs: dict[Cell, float] = {}
        for i in range(n):
            H_others = self._convolve(prefix[i], suffix[i + 1])
            for cell, h_cell_dict in h_cells[i].items():
                joint = self._convolve(h_cell_dict, H_others)
                probs[cell] = weighted_sum(joint) / total_weight

        if F > 0:
            off_num = 0.0
            for k, count in H_global.items():
                leftover = M - k
                if 0 <= leftover <= F:
                    off_num += count * math.comb(F, leftover) * leftover / F
            off_prob = off_num / total_weight
            for cell in off_frontier_cells:
                probs[cell] = off_prob

        return probs

    @staticmethod
    def _convolve(a: dict[int, int], b: dict[int, int]) -> dict[int, int]:
        result: dict[int, int] = {}
        for k1, v1 in a.items():
            for k2, v2 in b.items():
                k = k1 + k2
                result[k] = result.get(k, 0) + v1 * v2
        return result

    def _select_cell(
        self,
        probabilities: dict[Cell, float],
        frontier_cells: set[Cell],
    ) -> "tuple[str, int, int]":
        min_p = min(probabilities.values())
        candidates = [c for c, p in probabilities.items() if p == min_p]

        frontier_candidates = [c for c in candidates if c in frontier_cells]
        pool = frontier_candidates if frontier_candidates else candidates

        idx = int(self.rng.integers(len(pool)))
        r, c = pool[idx]
        return ("reveal", r, c)

    def _fallback(self, view: np.ndarray) -> "tuple[str, int, int]":
        constraints = self._build_constraints(view)
        if not constraints:
            return super()._fallback(view)

        frontier_cells, off_frontier_cells = self._find_frontier(view, constraints)
        components = self._split_components(frontier_cells, constraints)
        probabilities = self._compute_probabilities(
            components, off_frontier_cells, view
        )

        if not probabilities:
            return super()._fallback(view)

        self.last_probabilities = probabilities
        action = self._select_cell(probabilities, frontier_cells)
        _, r, c = action
        p = probabilities[(r, c)]
        region = "frontier" if (r, c) in frontier_cells else "off-frontier"
        self.last_reason = f"P(mine)={p:.1%} [{region}]"
        return action
