"""Unit tests for FinalAgent.

Focuses on the endgame solver: when few unrevealed cells remain, the
solver enumerates valid mine placements and should:
- reveal a certain-safe cell when one exists,
- flag a certain-mine cell when one exists,
- otherwise produce a probability-minimising guess.
"""
from __future__ import annotations

import numpy as np
import pytest

from agents.final_agent import FinalAgent
from minesweeper.board import CellState

U = int(CellState.UNREVEALED)
F = int(CellState.FLAGGED)
EPS = 1e-9


def make_agent(h: int, w: int, n_mines: int, seed: int = 0) -> FinalAgent:
    agent = FinalAgent(h, w, n_mines, rng=np.random.default_rng(seed))
    agent.reset()
    return agent


def test_endgame_flags_certain_mine():
    # 2x2 board, 1 mine. Three cells revealed. View:
    #   1 1
    #   1 U
    # Each "1" sees exactly one unrevealed neighbor: (1,1). All three
    # constraints force (1,1) to be the mine. Endgame solver should flag it.
    view = np.array([
        [1, 1],
        [1, U],
    ], dtype=np.int8)
    agent = make_agent(2, 2, 1)
    agent._last_view = view
    unrevealed = [(1, 1)]
    result = agent._endgame_solve(view, unrevealed)
    assert result is not None
    assert result == ("flag", 1, 1)


def test_endgame_reveals_certain_safe():
    # 2x2, 1 mine. View:
    #   0 1
    #   1 U
    # "0" at (0,0) implies (0,1),(1,0) safe (those are already revealed as 1,
    # which is consistent: their unrevealed neighbor is (1,1)).
    # "1" at (0,1) -> (1,1) is the mine.
    # "1" at (1,0) -> (1,1) is the mine.
    # Total mines = 1 -> (1,1) is the mine. So (1,1) is NOT safe, but with
    # only 1 cell to enumerate the solver would flag it.
    # Provide a board with at least one certain-safe cell:
    #
    # 2x3, 1 mine. View:
    #   1 1 U
    #   U U U
    # Hmm: "1" at (0,0) has unrevealed {(1,0),(1,1)} -> 1 mine.
    # "1" at (0,1) has unrevealed {(0,2),(1,0),(1,1),(1,2)} -> 1 mine.
    # Subset of first into second: (0,2) and (1,2) collectively contain 0
    # mines => both safe. But we want endgame to detect it directly.
    #
    # Reduce to fewer unrevealed cells with 2x2, 1 mine, view:
    #   2 U
    #   U U
    # "2" at (0,0): 3 unrevealed neighbors must contain 2 mines. But total
    # mines=1 -> infeasible. Try total=2:
    #
    # 2x2, 2 mines. View:
    #   2 U
    #   U U
    # 3 unrevealed cells with 2 mines among them. Solver enumerates choose(3,2)
    # =3 placements -> P=2/3 each, no certain safe/mine.
    #
    # 2x2, 1 mine, view:
    #   U U
    #   U 1
    # "1" at (1,1) has unrevealed {(0,0),(0,1),(1,0)} -> 1 mine; total = 1.
    # Each cell has P=1/3. No certain safe/mine -> falls into guess path.
    #
    # Build a setup with a certain-safe cell using a 1-1 subset:
    # 2x3, 1 mine, view:
    #   1 1 U
    #   . . .
    # where (1,0),(1,1),(1,2) are all revealed (say all 0). Then "1" at (0,0)
    # sees (0,1) -> mine. "1" at (0,1) sees (0,0),(0,2) -> 1 mine. Already
    # (0,1) is unrevealed too; let's restart with clear logic.
    #
    # Simplest: 1x3, 1 mine.
    #   1 U U
    # "1" at (0,0): unrevealed neighbor {(0,1)} -> mine. Total mines=1 so
    # (0,2) is safe. Endgame should reveal (0,2) as certain safe.
    view = np.array([[1, U, U]], dtype=np.int8)
    agent = make_agent(1, 3, 1)
    agent._last_view = view
    unrevealed = [(0, 1), (0, 2)]
    result = agent._endgame_solve(view, unrevealed)
    assert result is not None
    assert result == ("reveal", 0, 2)


def test_endgame_returns_none_for_infeasible():
    # 1x3, 2 mines but only 1 unrevealed cell -> infeasible.
    #   1 U 1
    # Wait, both 1s can be satisfied only if (0,1) is a mine. That's 1 mine
    # accounted for; remaining_mines=2 -> infeasible. Solver returns None.
    view = np.array([[1, U, 1]], dtype=np.int8)
    agent = make_agent(1, 3, 2)
    agent._last_view = view
    unrevealed = [(0, 1)]
    result = agent._endgame_solve(view, unrevealed)
    assert result is None


def test_endgame_skipped_when_too_many_unrevealed():
    # 40 unrevealed cells -> exceeds threshold of 36; _fallback must NOT
    # bump into the endgame regime. Build a 1x40 strip with no constraints.
    view = np.full((1, 40), U, dtype=np.int8)
    agent = make_agent(1, 40, 5)
    agent._last_view = view
    action = agent._fallback(view)
    assert action[0] in ("reveal", "flag")
    assert agent.endgame_calls == 0


def test_endgame_subset_safe_cell():
    # Classic 1-1 subset scenario, small enough for endgame solve to trigger.
    # 1x5, 1 mine, view:
    #   1 1 U U U
    # "1" at (0,0) -> 1 mine in {(0,1)} (since only (0,1) is unrevealed
    # neighbor) -- wait (0,1) is revealed as 1. Its unrevealed neighbors:
    # (0,1)'s neighbors are (0,0),(0,2). So "1" at (0,0): unrevealed neighbors
    # = {(0,1) is revealed, no others}. That's empty. Bad layout.
    #
    # Use 2x3:
    #   1 1 U
    #   U U U
    # Total mines = 1.
    # "1" at (0,0): unrevealed = {(1,0),(1,1)} -> 1 mine.
    # "1" at (0,1): unrevealed = {(0,2),(1,0),(1,1),(1,2)} -> 1 mine.
    # Subset: A ⊂ B with same count -> B \ A = {(0,2),(1,2)} has 0 mines.
    # Both safe. 5 unrevealed cells -> within endgame threshold.
    view = np.array([
        [1, 1, U],
        [U, U, U],
    ], dtype=np.int8)
    agent = make_agent(2, 3, 1)
    agent._last_view = view
    unrevealed = [(0, 2), (1, 0), (1, 1), (1, 2)]
    result = agent._endgame_solve(view, unrevealed)
    assert result is not None
    assert result[0] == "reveal"
    # The returned cell must be certain-safe: P=0.
    _, r, c = result
    assert (r, c) in {(0, 2), (1, 2)}


def test_act_first_move_center():
    view = np.full((9, 9), U, dtype=np.int8)
    agent = make_agent(9, 9, 10)
    action = agent.act(view)
    assert action == ("reveal", 4, 4)


def test_lookahead_disabled_increments_no_evals():
    # With lookahead off, the lookahead_evals counter must stay at zero even
    # when _select_cell is invoked with multiple candidates.
    view = np.full((5, 5), U, dtype=np.int8)
    view[0, 0] = 1
    agent = make_agent(5, 5, 3)
    agent.lookahead_enabled = False
    agent._last_view = view
    # Two cells with identical mine probability so the threshold filter keeps
    # both as candidates.
    probs = {(0, 1): 0.10, (4, 4): 0.10, (3, 3): 0.30}
    frontier = {(0, 1), (3, 3)}
    agent._select_cell(probs, frontier)
    assert agent.lookahead_evals == 0


def test_lookahead_enabled_runs_evals_and_records_metric():
    # With lookahead on, _select_cell should invoke _lookahead_ev for each of
    # the top-K candidates and record the per-candidate `lookahead_ev` in
    # last_candidate_ev.
    view = np.full((5, 5), U, dtype=np.int8)
    view[0, 0] = 1   # adjacent constraint so revealing (0,1) cascades info
    agent = make_agent(5, 5, 3)
    agent.lookahead_enabled = True
    agent._last_view = view
    probs = {(0, 1): 0.10, (4, 4): 0.10}
    frontier = {(0, 1)}
    agent._select_cell(probs, frontier)
    assert agent.lookahead_evals > 0
    # Each candidate that received lookahead must have the metric attached.
    for cell in probs:
        ev = agent.last_candidate_ev.get(cell)
        assert ev is not None
        assert "lookahead_ev" in ev, \
            f"missing lookahead_ev for {cell}: {ev!r}"


def test_lookahead_p_gap_skips_wide_spread():
    # Candidates with mine prob differing by more than lookahead_p_gap should
    # bypass lookahead entirely — raw risk already dominates.
    view = np.full((5, 5), U, dtype=np.int8)
    view[0, 0] = 1
    agent = make_agent(5, 5, 3)
    agent.lookahead_enabled = True
    agent.lookahead_p_gap = 0.01
    agent.risk_tolerance = 1.0  # let both into the candidate pool
    agent._last_view = view
    probs = {(0, 1): 0.05, (4, 4): 0.30}
    frontier = {(0, 1)}
    agent._select_cell(probs, frontier)
    assert agent.lookahead_evals == 0


def test_lookahead_prefers_cascade_cell():
    # Two candidates with identical mine probability. Cell B at (0,1) is
    # adjacent to a "1" hint so revealing it constrains the surrounding cells;
    # cell A at (4,4) is a lone corner whose reveal produces no new
    # deductions. Static EV alone may break ties either way (corner
    # centrality bonus vs frontier bonus); lookahead must shift the choice
    # toward the cell that actually unlocks information.
    view = np.full((5, 5), U, dtype=np.int8)
    view[0, 0] = 1
    probs = {(0, 1): 0.10, (4, 4): 0.10}
    frontier = {(0, 1)}

    # Disabled lookahead: deterministic seed; capture which cell static EV picks.
    agent_off = make_agent(5, 5, 3, seed=0)
    agent_off.lookahead_enabled = False
    agent_off._last_view = view
    pick_off = agent_off._select_cell(probs, frontier)

    # Enabled lookahead with the same seed.
    agent_on = make_agent(5, 5, 3, seed=0)
    agent_on.lookahead_enabled = True
    agent_on._last_view = view
    pick_on = agent_on._select_cell(probs, frontier)

    # The "informative" cell is (0,1) — it borders the "1" hint, so its
    # reveal triggers downstream CSP work. Lookahead should never penalise it.
    score_on_b = agent_on.last_candidate_scores[(0, 1)]
    score_on_a = agent_on.last_candidate_scores[(4, 4)]
    assert score_on_b >= score_on_a, (
        f"lookahead should not rank lone corner (4,4) above informative "
        f"frontier cell (0,1); got {score_on_a} >= {score_on_b}"
    )
    # And the picked cell must be the informative one.
    assert pick_on == ("reveal", 0, 1), f"expected (0,1), got {pick_on}"
    # Sanity: at least one candidate received the lookahead bonus.
    assert agent_on.lookahead_evals > 0


def test_endgame_force_mine_via_count_constraint():
    # 1x3, 1 mine total. View:
    #   U U 1
    # "1" at (0,2): unrevealed neighbor {(0,1)} -> mine. Then (0,0) is safe
    # by remaining_mines exhaustion. Endgame solver, given all unrevealed,
    # should pick certain-safe first.
    view = np.array([[U, U, 1]], dtype=np.int8)
    agent = make_agent(1, 3, 1)
    agent._last_view = view
    unrevealed = [(0, 0), (0, 1)]
    result = agent._endgame_solve(view, unrevealed)
    assert result is not None
    # Certain-safe (0,0) preferred over certain-mine (0,1).
    assert result == ("reveal", 0, 0)


def test_pending_safe_orders_high_cascade_first():
    # FinalAgent overrides _infer to apply cascade-priority ordering to
    # _pending_safe so the cell most likely to trigger flood-fill (more
    # unrevealed neighbours + central) gets revealed first.
    #
    # Layout (5x5). Two "0" revealed cells force several safes into
    # _pending_safe. Centred cells should end up at the END of the list
    # (LIFO pop -> highest priority first).
    view = np.full((5, 5), U, dtype=np.int8)
    view[0, 3] = 0
    view[3, 0] = 0
    agent = make_agent(5, 5, 1)
    agent._first_move = False
    agent._infer(view)
    safes = agent._pending_safe
    assert len(safes) >= 3, f"expected several safes, got {safes!r}"

    import math as _math
    h, w = 5, 5
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    max_dist = _math.hypot(cy, cx)

    def prio(cell):
        r, c = cell
        unrev = 0
        num = 0
        for dr, dc in [(-1, -1), (-1, 0), (-1, 1),
                       (0, -1),          (0, 1),
                       (1, -1),  (1, 0), (1, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                nv = int(view[nr, nc])
                if nv == U:
                    unrev += 1
                elif 0 <= nv <= 8:
                    num += 1
        centrality = 1.0 - _math.hypot(r - cy, c - cx) / max_dist
        return unrev * 3.0 + num * 1.0 + centrality * 0.5

    priorities = [prio(cell) for cell in safes]
    assert priorities == sorted(priorities), \
        f"_pending_safe not in ascending priority order: {priorities}"
    assert priorities[-1] == max(priorities)
