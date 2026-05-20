"""Final agent: strongest Minesweeper solver in the project.

This is where all optimizations live. CSPAgent and ProbabilityAgent are
intentionally kept as the textbook baselines they were originally written
as — pure inheritance baselines, easy to read. FinalAgent overrides their
methods with the optimized implementations:

R1 (speed)
  - vectorized `_build_constraints` (np.where instead of nested python loops)
  - vectorized `_find_frontier`
  - bitmask DFS `_enumerate_component_histograms` with constraint
    propagation (saturation + fill forcing); avoids the parent's per-cell
    list/set bookkeeping
  - numpy histogram convolution in `_compute_probabilities` (replaces the
    dict-based polynomial multiplication)

R2 (decision quality)
  - certainty extraction in `_fallback`: P(mine)=0 cells become deduced
    reveals (last_was_guess=False) and P(mine)=1 cells become deduced flags
  - per-component fallback: if one frontier component overflows the
    enumeration cap, its cells collapse into the off-frontier pool while
    the other components still contribute exact marginals
  - low-risk EV tie-break in `_select_cell` (frontier / centrality /
    p_zero proxy / unrevealed-neighbour count blended)
  - endgame regime: when few cells remain, swap to a larger solution cap

R3 (refinement)
  - cascade-priority `_order_pending` sort so likely-zero reveals trigger
    flood-fill first and unlock the most downstream deductions
  - bounded 1-ply lookahead in `_select_cell` (opt-in via lookahead_enabled)

The class is deliberately self-contained: dropping it into a project where
only CSPAgent / ProbabilityAgent exist as references should bring the
strongest behaviour without modifying the simpler parents.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from minesweeper.board import CellState

from .csp_agent import Constraint, NEIGHBORS
from .probability_agent import Component, ProbabilityAgent

Cell = tuple[int, int]


class FinalAgent(ProbabilityAgent):
    name = "final"

    # ---- Probability / endgame knobs ----
    endgame_threshold: int = 36
    max_endgame_solutions: int = 300_000
    max_component_solutions: int = 200_000
    risk_tolerance: float = 0.0

    # ---- Bounded 1-ply lookahead knobs ----
    # Default off: lookahead costs ~2-3x runtime on Expert without measurable
    # win-rate gain at n=2000. Kept as opt-in for ablation studies.
    lookahead_enabled: bool = False
    lookahead_top_k: int = 5
    lookahead_p_gap: float = 0.05
    lookahead_only_in_endgame: bool = False
    lookahead_static_weight: float = 1.0
    lookahead_weight: float = 1.0

    # ---- Static EV scoring weights (used in _select_cell) ----
    # Exposed as class attrs so the report can document the formula in one
    # place and ablation can override per-instance without subclassing.
    ev_mine_penalty: float = 30.0
    ev_zero_weight: float = 5.0
    ev_unrev_weight: float = 3.0
    ev_revealed_weight: float = 1.0
    ev_frontier_bonus: float = 10.0
    ev_centrality_weight: float = 2.0

    def __init__(
        self,
        height: int,
        width: int,
        n_mines: int,
        rng: np.random.Generator | None = None,
        endgame_threshold: int | None = None,
        max_endgame_solutions: int | None = None,
        max_component_solutions: int | None = None,
        risk_tolerance: float | None = None,
        lookahead_enabled: bool | None = None,
        lookahead_top_k: int | None = None,
        lookahead_p_gap: float | None = None,
        lookahead_only_in_endgame: bool | None = None,
        lookahead_static_weight: float | None = None,
        lookahead_weight: float | None = None,
    ):
        super().__init__(height, width, n_mines, rng=rng)
        cls = type(self)
        self.endgame_threshold = (
            cls.endgame_threshold if endgame_threshold is None else endgame_threshold
        )
        self.max_endgame_solutions = (
            cls.max_endgame_solutions
            if max_endgame_solutions is None else max_endgame_solutions
        )
        self.max_component_solutions = (
            cls.max_component_solutions
            if max_component_solutions is None else max_component_solutions
        )
        self.risk_tolerance = (
            cls.risk_tolerance if risk_tolerance is None else risk_tolerance
        )
        self.lookahead_enabled = (
            cls.lookahead_enabled if lookahead_enabled is None else lookahead_enabled
        )
        self.lookahead_top_k = (
            cls.lookahead_top_k if lookahead_top_k is None else lookahead_top_k
        )
        self.lookahead_p_gap = (
            cls.lookahead_p_gap if lookahead_p_gap is None else lookahead_p_gap
        )
        self.lookahead_only_in_endgame = (
            cls.lookahead_only_in_endgame
            if lookahead_only_in_endgame is None else lookahead_only_in_endgame
        )
        self.lookahead_static_weight = (
            cls.lookahead_static_weight
            if lookahead_static_weight is None else lookahead_static_weight
        )
        self.lookahead_weight = (
            cls.lookahead_weight if lookahead_weight is None else lookahead_weight
        )

    def reset(self) -> None:
        super().reset()
        self._last_view: np.ndarray | None = None
        self.last_candidate_scores: dict[Cell, float] = {}
        self.last_candidate_ev: dict[Cell, dict[str, float]] = {}
        self.endgame_calls: int = 0
        self.endgame_aborts: int = 0
        self.lookahead_evals: int = 0
        self._last_compute_aborted: bool = False

    def act(self, view: np.ndarray):
        self._last_view = view
        self.last_was_guess = False
        self.last_probabilities = {}
        self.last_candidate_scores = {}
        self.last_candidate_ev = {}
        # Skip ProbabilityAgent.act (its job is just to clear
        # last_probabilities, which we already did) and use CSPAgent.act
        # directly so its deduce-loop calls our overridden _fallback.
        return super(ProbabilityAgent, self).act(view)

    # ------------------------------------------------------------------
    # R1 / R2: vectorized constraint + frontier construction
    # ------------------------------------------------------------------
    def _build_constraints(self, view: np.ndarray) -> list[Constraint]:
        """Override CSPAgent._build_constraints with a numpy-vectorised scan
        of revealed numeric cells. Identical output, faster on large boards."""
        unrev_val = int(CellState.UNREVEALED)
        flagged_val = int(CellState.FLAGGED)
        rs, cs = np.where((view >= 0) & (view <= 8))
        out: list[Constraint] = []
        H, W = self.h, self.w
        for r, c in zip(rs.tolist(), cs.tolist()):
            v = int(view[r, c])
            unrevealed: list[Cell] = []
            flagged = 0
            for dr, dc in NEIGHBORS:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                nv = int(view[nr, nc])
                if nv == unrev_val:
                    unrevealed.append((nr, nc))
                elif nv == flagged_val:
                    flagged += 1
            if unrevealed:
                out.append(Constraint(frozenset(unrevealed), v - flagged))
        return out

    def _find_frontier(
        self, view: np.ndarray, constraints: list[Constraint]
    ) -> tuple[set[Cell], set[Cell]]:
        """Vectorised frontier / off-frontier split (R2)."""
        frontier: set[Cell] = set()
        for con in constraints:
            frontier.update(con.cells)
        rs, cs = np.where(view == int(CellState.UNREVEALED))
        all_unrev = {(int(r), int(c)) for r, c in zip(rs.tolist(), cs.tolist())}
        off_frontier = all_unrev - frontier
        return frontier, off_frontier

    # ------------------------------------------------------------------
    # R3: cascade-priority ordering of deduced safes / mines
    # ------------------------------------------------------------------
    def _infer(self, view: np.ndarray) -> None:
        """Run CSPAgent._infer, then reorder the deduced queues by cascade
        priority so likely-zero reveals trigger flood-fill first.

        `_pending_safe` and `_pending_flag` are consumed via list.pop()
        (LIFO). We sort ASCENDING by priority so the LAST element popped
        is the HIGHEST priority cell.
        """
        super()._infer(view)
        self._pending_safe = self._order_pending(view, self._pending_safe)
        self._pending_flag = self._order_pending(view, self._pending_flag)

    def _order_pending(self, view: np.ndarray, cells: list[Cell]) -> list[Cell]:
        if len(cells) <= 1:
            return cells
        unrev_val = int(CellState.UNREVEALED)
        H, W = self.h, self.w
        cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
        max_dist = math.hypot(cy, cx) or 1.0

        def priority(cell: Cell) -> float:
            r, c = cell
            unrev_n = 0
            revealed_num = 0
            for dr, dc in NEIGHBORS:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                nv = int(view[nr, nc])
                if nv == unrev_val:
                    unrev_n += 1
                elif 0 <= nv <= 8:
                    revealed_num += 1
            centrality = 1.0 - math.hypot(r - cy, c - cx) / max_dist
            return unrev_n * 3.0 + revealed_num * 1.0 + centrality * 0.5

        return sorted(cells, key=priority)

    # ------------------------------------------------------------------
    # R1: bitmask DFS enumeration with constraint propagation
    # ------------------------------------------------------------------
    def _enumerate_component_histograms(
        self, component: Component
    ) -> Optional[tuple[list[Cell], np.ndarray, list[np.ndarray]]]:
        """Bitmask DFS over a single connected component. Returns
            (ordered_cells, h_total, h_per_cell)
        where:
            h_total[k]      = #(valid assignments with exactly k mines)
            h_per_cell[j][k] = #(valid assignments with k mines AND cell j is mine)

        Returns None if enumeration exceeded `max_component_solutions`
        (the caller treats this as an aborted component and folds those
        cells into the off-frontier pool).
        """
        cells, constraints = component
        ordered_cells: list[Cell] = sorted(cells)
        n = len(ordered_cells)
        if n == 0:
            return ordered_cells, np.zeros(1, dtype=np.float64), []
        cell_to_idx = {cell: i for i, cell in enumerate(ordered_cells)}

        # Encode each constraint as (cell_mask, required_mines).
        con_masks: list[tuple[int, int]] = []
        for con in constraints:
            mask = 0
            for c in con.cells:
                mask |= 1 << cell_to_idx[c]
            con_masks.append((mask, con.n_mines))

        # cell_to_cons[i] = constraint indices touching cell i (used to
        # decide which constraints to re-check after a forced assignment).
        cell_to_cons: list[list[int]] = [[] for _ in range(n)]
        for ci, (mask, _) in enumerate(con_masks):
            m = mask
            while m:
                lo = (m & -m).bit_length() - 1
                cell_to_cons[lo].append(ci)
                m &= m - 1

        full_mask = (1 << n) - 1
        max_solutions = int(self.max_component_solutions)
        solutions_count = [0]
        aborted = [False]
        h_total = np.zeros(n + 1, dtype=np.float64)
        h_per_cell = np.zeros((n, n + 1), dtype=np.float64)

        def record(assignment: int, mines_so_far: int) -> None:
            h_total[mines_so_far] += 1.0
            a = assignment
            while a:
                lo = (a & -a).bit_length() - 1
                h_per_cell[lo, mines_so_far] += 1.0
                a &= a - 1
            solutions_count[0] += 1
            if solutions_count[0] >= max_solutions:
                aborted[0] = True

        def propagate(
            assignment: int, undecided_mask: int, mines_so_far: int,
            touched_cell: int,
        ):
            """Validate constraints touching `touched_cell` and force any
            cells they determine. Returns updated (assignment,
            undecided_mask, mines_so_far) on success, None if infeasible."""
            queue = list(cell_to_cons[touched_cell])
            in_queue = {ci for ci in queue}
            while queue:
                ci = queue.pop()
                in_queue.discard(ci)
                cmask, required = con_masks[ci]
                decided_mines = bin(assignment & cmask).count("1")
                undecided_in_con = bin(cmask & undecided_mask).count("1")
                if decided_mines > required:
                    return None
                if decided_mines + undecided_in_con < required:
                    return None
                if undecided_in_con == 0:
                    continue
                if decided_mines == required:
                    # Saturation: all remaining cells in this constraint are safe.
                    cleared = cmask & undecided_mask
                    undecided_mask ^= cleared
                    # Re-check other constraints touching the cleared cells.
                    m = cleared
                    while m:
                        lo = (m & -m).bit_length() - 1
                        for cj in cell_to_cons[lo]:
                            if cj != ci and cj not in in_queue:
                                in_queue.add(cj); queue.append(cj)
                        m &= m - 1
                elif decided_mines + undecided_in_con == required:
                    # Fill: all remaining undecided cells in this constraint
                    # are mines.
                    filled = cmask & undecided_mask
                    assignment |= filled
                    mines_so_far += bin(filled).count("1")
                    undecided_mask ^= filled
                    m = filled
                    while m:
                        lo = (m & -m).bit_length() - 1
                        for cj in cell_to_cons[lo]:
                            if cj != ci and cj not in in_queue:
                                in_queue.add(cj); queue.append(cj)
                        m &= m - 1
            return assignment, undecided_mask, mines_so_far

        def dfs(i: int, assignment: int, undecided_mask: int,
                mines_so_far: int) -> None:
            if aborted[0]:
                return
            # Advance past any cells that propagation already decided.
            while i < n and not ((undecided_mask >> i) & 1):
                i += 1
            if i == n:
                record(assignment, mines_so_far)
                return
            bit = 1 << i
            # Branch 1: cell i is safe (bit stays 0).
            res = propagate(assignment, undecided_mask ^ bit, mines_so_far, i)
            if res is not None:
                dfs(i + 1, *res)
                if aborted[0]:
                    return
            # Branch 2: cell i is a mine.
            res = propagate(assignment | bit, undecided_mask ^ bit,
                            mines_so_far + 1, i)
            if res is not None:
                dfs(i + 1, *res)

        dfs(0, 0, full_mask, 0)
        if aborted[0]:
            return None
        if h_total.sum() == 0:
            return None
        return ordered_cells, h_total, [h_per_cell[j] for j in range(n)]

    # ------------------------------------------------------------------
    # R1 + R2: numpy convolution + per-component fallback
    # ------------------------------------------------------------------
    def _compute_probabilities(
        self,
        components: list[Component],
        off_frontier_cells: set[Cell],
        view: np.ndarray,
    ) -> dict[Cell, float]:
        """Optimized override of ProbabilityAgent._compute_probabilities.

        Differences from the parent:
        - Per-component enumeration uses the bitmask DFS via
          `_enumerate_component_histograms` (returns numpy histograms
          directly).
        - Per-component fallback (R2 MAJOR-3): if one component aborts
          enumeration, its cells fold into off-frontier rather than
          poisoning the entire turn.
        - Polynomial multiplication uses `np.convolve` over float64
          int-valued arrays.
        - Off-frontier weighting normalised by `max(C(F, leftover))` to
          keep ratios in float64 even for astronomical binomials.
        """
        self._last_compute_aborted = False
        flagged_count = int((view == int(CellState.FLAGGED)).sum())
        M = self.n_mines - flagged_count

        comp_h_totals: list[np.ndarray] = []
        comp_h_per_cell: list[list[np.ndarray]] = []
        comp_cells_list: list[list[Cell]] = []
        aborted_cells: set[Cell] = set()

        for comp in components:
            result = self._enumerate_component_histograms(comp)
            if result is None:
                self._last_compute_aborted = True
                aborted_cells.update(comp[0])
                continue
            ordered, h_total, h_per_cell = result
            comp_h_totals.append(h_total)
            comp_h_per_cell.append(h_per_cell)
            comp_cells_list.append(ordered)

        if aborted_cells:
            off_frontier_cells = set(off_frontier_cells) | aborted_cells
        F = len(off_frontier_cells)

        n = len(comp_h_totals)
        identity = np.array([1.0], dtype=np.float64)
        prefix: list[np.ndarray] = [identity]
        for h in comp_h_totals:
            prefix.append(np.convolve(prefix[-1], h))
        suffix: list[np.ndarray] = [identity] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix[i] = np.convolve(suffix[i + 1], comp_h_totals[i])

        H_global = prefix[n]
        max_len = len(H_global) + max(
            (len(h) for hpc in comp_h_per_cell for h in hpc), default=0
        )

        # weights_full[k] = C(F, M - k), normalised by its max so the
        # final ratios stay numerically tame.
        weights_full = np.zeros(max_len, dtype=np.float64)
        max_w = 0.0
        for k in range(max_len):
            leftover = M - k
            if 0 <= leftover <= F:
                c = float(math.comb(F, leftover))
                weights_full[k] = c
                if c > max_w:
                    max_w = c
        if max_w > 0:
            weights_full /= max_w

        def weighted_sum(hist: np.ndarray) -> float:
            length = len(hist)
            if length == 0:
                return 0.0
            return float(np.dot(hist, weights_full[:length]))

        total_weight = weighted_sum(H_global)
        if total_weight == 0:
            return {}

        probs: dict[Cell, float] = {}
        for i in range(n):
            H_others = np.convolve(prefix[i], suffix[i + 1])
            cells_i = comp_cells_list[i]
            per_cell = comp_h_per_cell[i]
            for j, cell in enumerate(cells_i):
                joint = np.convolve(per_cell[j], H_others)
                probs[cell] = weighted_sum(joint) / total_weight

        if F > 0:
            length = len(H_global)
            ks = np.arange(length, dtype=np.float64)
            leftover = M - ks
            mask = (leftover >= 0) & (leftover <= F)
            leftover_safe = np.where(mask, leftover, 0.0)
            off_num = float(
                np.dot(H_global * weights_full[:length], leftover_safe)
            ) / F
            off_prob = off_num / total_weight
            for cell in off_frontier_cells:
                probs[cell] = off_prob

        return probs

    # ------------------------------------------------------------------
    # R2: certainty extraction + R3 endgame regime
    # ------------------------------------------------------------------
    def _fallback(self, view: np.ndarray):
        """Run the probability pipeline with FinalAgent's optimised
        helpers, extract P=0 / P=1 certainties as deductions, and only
        enter the EV / lookahead guess path if no certainty was found.
        """
        # Endgame regime: temporarily raise the per-component solution cap
        # so decisive late-game moves get the full budget.
        unrevealed_count = int((view == int(CellState.UNREVEALED)).sum())
        in_endgame_regime = unrevealed_count <= self.endgame_threshold
        old_cap = None
        if in_endgame_regime:
            self.endgame_calls += 1
            old_cap = self.max_component_solutions
            self.max_component_solutions = self.max_endgame_solutions

        try:
            constraints = self._build_constraints(view)
            if not constraints:
                return super(ProbabilityAgent, self)._fallback(view)

            frontier_cells, off_frontier_cells = self._find_frontier(view, constraints)
            components = self._split_components(frontier_cells, constraints)
            probabilities = self._compute_probabilities(
                components, off_frontier_cells, view
            )
            if not probabilities:
                return super(ProbabilityAgent, self)._fallback(view)

            self.last_probabilities = probabilities

            # R2 certainty extraction: P=0 cells are deduced safes,
            # P=1 cells are deduced mines. They are NOT guesses.
            certain_safes: list[Cell] = []
            certain_mines: list[Cell] = []
            for cell, p in probabilities.items():
                if p <= 1e-12:
                    certain_safes.append(cell)
                elif p >= 1.0 - 1e-12:
                    certain_mines.append(cell)

            if certain_safes:
                ordered = self._order_pending(view, certain_safes)
                r, c = ordered[-1]
                self._pending_safe.extend(ordered[:-1])
                self.last_was_guess = False
                self.last_reason = "probability-deduced safe"
                return ("reveal", r, c)
            if certain_mines:
                ordered = self._order_pending(view, certain_mines)
                r, c = ordered[-1]
                self._pending_flag.extend(ordered[:-1])
                self.last_was_guess = False
                self.last_reason = "probability-deduced mine"
                return ("flag", r, c)

            # Genuine guess: low-risk EV + optional lookahead.
            self.last_was_guess = True
            action = self._select_cell(probabilities, frontier_cells)
            _, r, c = action
            p = probabilities[(r, c)]
            region = "frontier" if (r, c) in frontier_cells else "off-frontier"
            self.last_reason = f"P(mine)={p:.1%} [{region}]"
            return action
        finally:
            if old_cap is not None:
                self.max_component_solutions = old_cap
            # Single source of truth for endgame_aborts: count once per
            # _fallback call when we were in the endgame regime and at least
            # one component aborted enumeration. Works for both early-return
            # (no probabilities) and normal-completion paths.
            if in_endgame_regime and self._last_compute_aborted:
                self.endgame_aborts += 1

    # ------------------------------------------------------------------
    # Test-only shim. Production endgame uses _fallback().
    # ------------------------------------------------------------------
    def _endgame_solve(self, view: np.ndarray, unrevealed: list[Cell]):
        """Test-only API. Restricts the standard probability pipeline to
        the supplied unrevealed cells and returns the first certain
        safe / certain mine, or a low-risk guess. Production endgame
        does not call this — it goes through `_fallback` with a swapped
        solution cap.
        """
        unrevealed_set = set(unrevealed)
        constraints = self._build_constraints(view)
        restricted: list[Constraint] = []
        for con in constraints:
            cells = con.cells & unrevealed_set
            if cells:
                restricted.append(Constraint(frozenset(cells), con.n_mines))

        frontier_cells, _ = self._find_frontier(view, restricted)
        off_frontier = unrevealed_set - frontier_cells
        components = self._split_components(frontier_cells, restricted)
        probabilities = self._compute_probabilities(components, off_frontier, view)
        if not probabilities:
            return None

        probabilities = {
            cell: p for cell, p in probabilities.items() if cell in unrevealed_set
        }
        if not probabilities:
            return None

        certain_safe: Cell | None = None
        certain_mine: Cell | None = None
        for cell, p in probabilities.items():
            if p <= 1e-12 and certain_safe is None:
                certain_safe = cell
            elif p >= 1.0 - 1e-12 and certain_mine is None:
                certain_mine = cell
        if certain_safe is not None:
            self.last_reason = "endgame: certain safe"
            return ("reveal", *certain_safe)
        if certain_mine is not None:
            self.last_reason = "endgame: certain mine"
            return ("flag", *certain_mine)

        self.last_was_guess = True
        self.last_probabilities = probabilities
        action = self._select_cell(probabilities, frontier_cells)
        _, r, c = action
        p = probabilities[(r, c)]
        self.last_reason = f"endgame: P(mine)={p:.1%}"
        return action

    # ------------------------------------------------------------------
    # R2 EV tie-break + R3 bounded 1-ply lookahead
    # ------------------------------------------------------------------
    def _select_cell(self, probabilities, frontier_cells):
        """Select a low-risk cell using a local expected-value tie-break,
        optionally refined by 1-ply lookahead."""
        eps = 1e-12
        min_p = min(probabilities.values())
        threshold = min_p + self.risk_tolerance
        candidates = [c for c, p in probabilities.items() if p <= threshold + eps]

        if len(candidates) == 1:
            r, c = candidates[0]
            return ("reveal", r, c)

        view = self._last_view
        unrevealed = int(CellState.UNREVEALED)
        avg_p = sum(probabilities.values()) / len(probabilities)
        max_dist = math.hypot(self.h / 2, self.w / 2) or 1.0

        scored: list[tuple[float, int, int]] = []
        candidate_scores: dict[Cell, float] = {}
        candidate_ev: dict[Cell, dict[str, float]] = {}

        for r, c in candidates:
            p_mine = probabilities[(r, c)]
            unrevealed_neighbors = 0
            revealed_numeric = 0
            local_safe_product = 1.0
            for dr, dc in NEIGHBORS:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < self.h and 0 <= nc < self.w):
                    continue
                if view is None:
                    continue
                nv = int(view[nr, nc])
                if 0 <= nv <= 8:
                    revealed_numeric += 1
                elif nv == unrevealed:
                    unrevealed_neighbors += 1
                    local_p = probabilities.get((nr, nc), avg_p)
                    local_safe_product *= max(0.0, min(1.0, 1.0 - local_p))
            p_zero = local_safe_product if unrevealed_neighbors else 0.0
            risk_score = -p_mine * self.ev_mine_penalty
            zero_score = p_zero * self.ev_zero_weight
            unknown_score = unrevealed_neighbors * self.ev_unrev_weight
            revealed_score = revealed_numeric * self.ev_revealed_weight
            frontier_score = self.ev_frontier_bonus if (r, c) in frontier_cells else 0.0
            dist = math.hypot(r - self.h / 2, c - self.w / 2)
            centrality_score = (1 - dist / max_dist) * self.ev_centrality_weight
            ev = (
                risk_score + zero_score + unknown_score
                + revealed_score + frontier_score + centrality_score
            )
            scored.append((ev, r, c))
            candidate_scores[(r, c)] = ev
            candidate_ev[(r, c)] = {
                "score": ev,
                "mine_prob": p_mine,
                "zero_prob": p_zero,
                "unrevealed_neighbors": float(unrevealed_neighbors),
                "revealed_numeric_neighbors": float(revealed_numeric),
                "risk_score": risk_score,
                "zero_score": zero_score,
                "unknown_score": unknown_score,
                "revealed_score": revealed_score,
                "frontier_score": frontier_score,
                "centrality_score": centrality_score,
            }

        if self._lookahead_active(candidates, probabilities):
            scored.sort(key=lambda x: -x[0])
            top_k = min(self.lookahead_top_k, len(scored))
            top_cells = [(r, c) for _, r, c in scored[:top_k]]
            for r, c in top_cells:
                la = self._lookahead_ev(r, c, probabilities, view)
                if la is None:
                    continue
                p_mine = probabilities[(r, c)]
                blended = (
                    self.lookahead_static_weight * candidate_scores[(r, c)]
                    + self.lookahead_weight * la * (1.0 - p_mine)
                )
                candidate_scores[(r, c)] = blended
                candidate_ev[(r, c)]["lookahead_ev"] = la
                candidate_ev[(r, c)]["score"] = blended
            scored = [(candidate_scores[c], c[0], c[1]) for c in candidates]

        self.last_candidate_scores = candidate_scores
        self.last_candidate_ev = candidate_ev

        scored.sort(key=lambda x: -x[0])
        best_score = scored[0][0]
        best = [(r, c) for s, r, c in scored if abs(s - best_score) < 1e-9]
        idx = int(self.rng.integers(len(best)))
        r, c = best[idx]
        return ("reveal", r, c)

    def _lookahead_active(
        self,
        candidates: list[Cell],
        probabilities: dict[Cell, float],
    ) -> bool:
        if not self.lookahead_enabled:
            return False
        if len(candidates) < 2:
            return False
        if self._last_view is None:
            return False
        if self.lookahead_only_in_endgame:
            n_unrev = int((self._last_view == int(CellState.UNREVEALED)).sum())
            if n_unrev > self.endgame_threshold:
                return False
        ps = [probabilities[c] for c in candidates]
        if max(ps) - min(ps) > self.lookahead_p_gap:
            return False
        return True

    def _lookahead_ev(
        self,
        r: int,
        c: int,
        probabilities: dict[Cell, float],
        view: np.ndarray,
    ) -> Optional[float]:
        """Estimate expected forced-deduction count if we reveal (r, c)."""
        if view is None:
            return None
        UNREV = int(CellState.UNREVEALED)
        H, W = self.h, self.w
        unrev_neighbours: list[Cell] = []
        flagged_neighbours = 0
        for dr, dc in NEIGHBORS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            nv = int(view[nr, nc])
            if nv == UNREV:
                unrev_neighbours.append((nr, nc))
            elif nv == int(CellState.FLAGGED):
                flagged_neighbours += 1
        if not unrev_neighbours:
            return 0.0

        # P(neighbour is a mine | this cell is safe) approximated as the
        # neighbour's marginal. Treats neighbours as independent (loose).
        ps = [probabilities.get(n, 0.0) for n in unrev_neighbours]
        # Poisson-binomial pmf over k=number-of-mines-in-neighbours.
        pmf = np.array([1.0], dtype=np.float64)
        for p in ps:
            pmf = np.convolve(pmf, np.array([1 - p, p], dtype=np.float64))

        ev = 0.0
        for k, prob_k in enumerate(pmf):
            if prob_k <= 0:
                continue
            value = flagged_neighbours + k
            forced = self._forced_deductions_after_reveal(view, r, c, value)
            ev += prob_k * forced
            self.lookahead_evals += 1
        return ev

    def _forced_deductions_after_reveal(
        self,
        view: np.ndarray,
        r: int,
        c: int,
        value: int,
    ) -> int:
        """Count safes / mines forced when (r,c) is hypothetically revealed
        with the given numeric value. Uses a hypothetical view + one pass
        of _build_constraints + trivial saturation / fill rules.
        """
        hyp = view.copy()
        hyp[r, c] = value
        constraints = self._build_constraints(hyp)
        if not constraints:
            return 0
        forced_safe: set[Cell] = set()
        forced_mine: set[Cell] = set()
        for con in constraints:
            if con.n_mines == 0:
                forced_safe.update(con.cells)
            elif con.n_mines == len(con.cells):
                forced_mine.update(con.cells)
        return len(forced_safe) + len(forced_mine)
