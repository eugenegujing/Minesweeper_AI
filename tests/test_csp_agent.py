"""Unit tests for CSPAgent.

Tests hand-crafted board views to verify the CSP rules:
- Trivial rule "n_mines == 0": all neighbors safe.
- Trivial rule "n_mines == len(cells)": all neighbors mines.
- Subset reasoning: A strict subset of B with same/different counts derives info.
"""
from __future__ import annotations

import numpy as np

from agents.csp_agent import CSPAgent
from minesweeper.board import CellState

U = int(CellState.UNREVEALED)
F = int(CellState.FLAGGED)


def make_agent(h: int, w: int, n_mines: int, seed: int = 0) -> CSPAgent:
    agent = CSPAgent(h, w, n_mines, rng=np.random.default_rng(seed))
    agent.reset()
    # Skip "first move" branch so .act runs the real inference path.
    agent._first_move = False
    return agent


def test_trivial_zero_marks_all_neighbors_safe():
    # 3x3 board. Center "0" means every neighbor is safe. Suppose we've already
    # revealed (1,1)=0 and one of its corners. Remaining unrevealed neighbors
    # must all be safe.
    view = np.array([
        [U, U, U],
        [U, 0, U],
        [U, U, U],
    ], dtype=np.int8)
    agent = make_agent(3, 3, 0)
    agent._infer(view)
    safe = set(agent._pending_safe)
    assert safe == {(0, 0), (0, 1), (0, 2),
                    (1, 0),         (1, 2),
                    (2, 0), (2, 1), (2, 2)}
    assert agent._pending_flag == []


def test_trivial_full_count_flags_all_neighbors():
    # Center "8" with all 8 corners unrevealed -> all 8 are mines.
    view = np.array([
        [U, U, U],
        [U, 8, U],
        [U, U, U],
    ], dtype=np.int8)
    agent = make_agent(3, 3, 8)
    agent._infer(view)
    mines = set(agent._pending_flag)
    assert mines == {(0, 0), (0, 1), (0, 2),
                     (1, 0),         (1, 2),
                     (2, 0), (2, 1), (2, 2)}
    assert agent._pending_safe == []


def test_count_minus_flagged_reduces_constraint():
    # A "2" cell with two unrevealed neighbors and one already-flagged neighbor:
    # effective count is 2 - 1 = 1, so the two unrevealed must contain exactly 1
    # mine. Not enough alone to deduce a certain cell. But combine with a "1"
    # whose only unrevealed neighbor is one of those: that cell is the mine.
    #
    #   F 2 1
    #   U U .
    # (column 2 row 1 is revealed somehow; here we put a "1" at (0,2) whose
    # unrevealed neighbor set will pin things down.)
    #
    # Layout 2x3:
    #   F 2 .         where '.' = revealed 1 with no remaining unrevealed.
    #   U U U
    # The "2" at (0,1) has flagged (0,0) + unrevealed (1,0),(1,1),(1,2). So
    # effective: 1 mine among {(1,0),(1,1),(1,2)}. Not determinable alone.
    #
    # Use subset reasoning instead — see next test.
    view = np.array([
        [F, 2, U],
        [U, U, U],
    ], dtype=np.int8)
    agent = make_agent(2, 3, 2)
    constraints = agent._build_constraints(view)
    # The "2" at (0,1) sees flagged (0,0), so effective n_mines = 1.
    target = [c for c in constraints
              if c.cells == frozenset({(0, 2), (1, 0), (1, 1), (1, 2)})]
    assert len(target) == 1
    assert target[0].n_mines == 1


def test_subset_reasoning_isolates_difference():
    # Classic 1-2 wall pattern. View:
    #   . 1 2 .
    #   U U U U
    # All four bottom cells unrevealed.
    # The "1" at (0,1) -> 1 mine among {(1,0),(1,1),(1,2)}.
    # The "2" at (0,2) -> 2 mines among {(1,1),(1,2),(1,3)}.
    # No strict subset directly. But take the "1"'s set A = {(1,0),(1,1),(1,2)}
    # and the "2"'s set B = {(1,1),(1,2),(1,3)}, neither is subset.
    #
    # Use simpler subset case: 1 at (0,0) constrains {(0,1),(1,0),(1,1)}=1; a
    # second 1 at (0,2) (off the board here, redo) — let's do a clear subset.
    #
    # Board 2x3:
    #   1 1 .
    #   U U U
    # "1" at (0,0): unrevealed neighbors {(0,1)??no, (0,1)=1 revealed; (1,0),(1,1)} => 1 mine in {(1,0),(1,1)}.
    # "1" at (0,1): unrevealed neighbors {(1,0),(1,1),(1,2)} => 1 mine in {(1,0),(1,1),(1,2)}.
    # A = {(1,0),(1,1)} ⊂ B = {(1,0),(1,1),(1,2)}, n_A=1, n_B=1.
    # Therefore B \ A = {(1,2)} has 0 mines -> safe.
    view = np.array([
        [1, 1, U],
        [U, U, U],
    ], dtype=np.int8)
    agent = make_agent(2, 3, 1)
    agent._infer(view)
    assert (1, 2) in agent._pending_safe


def test_first_move_returns_center():
    view = np.full((9, 9), U, dtype=np.int8)
    agent = CSPAgent(9, 9, 10, rng=np.random.default_rng(0))
    agent.reset()
    action = agent.act(view)
    assert action == ("reveal", 4, 4)


def test_fallback_marks_as_guess():
    # Totally blank middle-of-game view -> no constraints -> fallback. The
    # agent should mark last_was_guess=True.
    view = np.full((3, 3), U, dtype=np.int8)
    agent = make_agent(3, 3, 1)
    action = agent.act(view)
    assert action[0] == "reveal"
    assert agent.last_was_guess is True
