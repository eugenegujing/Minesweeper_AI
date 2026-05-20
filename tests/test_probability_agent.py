"""Unit tests for ProbabilityAgent.

Constructs hand-crafted board views with analytically known mine
probabilities and verifies _compute_probabilities matches within 1e-9.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from agents.probability_agent import ProbabilityAgent
from minesweeper.board import CellState

U = int(CellState.UNREVEALED)
F = int(CellState.FLAGGED)
EPS = 1e-9


def make_agent(h: int, w: int, n_mines: int, seed: int = 0) -> ProbabilityAgent:
    agent = ProbabilityAgent(h, w, n_mines, rng=np.random.default_rng(seed))
    agent.reset()
    return agent


def all_probs(agent: ProbabilityAgent, view: np.ndarray) -> dict:
    """Run the probability pipeline directly and return its probability map."""
    constraints = agent._build_constraints(view)
    frontier, off = agent._find_frontier(view, constraints)
    components = agent._split_components(frontier, constraints)
    return agent._compute_probabilities(components, off, view)


def test_single_one_constraint_three_neighbors():
    # 3x3 board with one mine total. View:
    #   1 . .
    #   . . .
    #   . . .
    # The "1" at (0,0) constrains its three unrevealed neighbors
    # {(0,1),(1,0),(1,1)} to contain exactly 1 mine.
    # Since total mines = 1, the off-frontier (5 cells) has 0 mines.
    # P(mine) for each frontier cell = 1/3, off-frontier = 0.
    view = np.array([
        [1, U, U],
        [U, U, U],
        [U, U, U],
    ], dtype=np.int8)
    agent = make_agent(3, 3, 1)
    probs = all_probs(agent, view)

    frontier_cells = {(0, 1), (1, 0), (1, 1)}
    off_cells = {(0, 2), (1, 2), (2, 0), (2, 1), (2, 2)}

    for cell in frontier_cells:
        assert probs[cell] == pytest.approx(1 / 3, abs=EPS)
    for cell in off_cells:
        assert probs[cell] == pytest.approx(0.0, abs=EPS)


def test_single_one_two_neighbors_half_each():
    # 2x2 board with 1 mine, both top cells revealed as "1", both bottom cells
    # unrevealed.
    #   1 1
    #   U U
    # Each "1" constrains its two unrevealed neighbors (they share {(1,0),
    # (1,1)}) to contain exactly 1 mine. With 1 mine total, P=0.5 each.
    view = np.array([
        [1, 1],
        [U, U],
    ], dtype=np.int8)
    agent = make_agent(2, 2, 1)
    probs = all_probs(agent, view)

    assert probs[(1, 0)] == pytest.approx(0.5, abs=EPS)
    assert probs[(1, 1)] == pytest.approx(0.5, abs=EPS)


def test_off_frontier_probability_weighted_correctly():
    # 3x3, 2 mines total. View:
    #   1 0 .
    #   . . .
    #   . . .
    # "1" at (0,0): 1 mine among {(1,0),(1,1)} (since (0,1)=0 reveals (0,2) is
    # safe too, but we put '.' = U for the rest to keep them off-frontier).
    # "0" at (0,1) forces (0,2),(1,0),(1,1),(1,2) safe -> contradicts "1".
    # So scrap. Easier: do NOT include the "0".
    #
    # 3x3, total mines=2:
    #   1 . .
    #   . . .
    #   . . .
    # Frontier = {(0,1),(1,0),(1,1)} with 1 mine; off-frontier = 5 cells with
    # the remaining 1 mine.
    # P(mine | frontier cell) = 1/3.
    # P(mine | off-frontier cell) = 1/5.
    view = np.array([
        [1, U, U],
        [U, U, U],
        [U, U, U],
    ], dtype=np.int8)
    agent = make_agent(3, 3, 2)
    probs = all_probs(agent, view)

    frontier = {(0, 1), (1, 0), (1, 1)}
    off = {(0, 2), (1, 2), (2, 0), (2, 1), (2, 2)}

    for cell in frontier:
        assert probs[cell] == pytest.approx(1 / 3, abs=EPS), f"{cell}: {probs[cell]}"
    for cell in off:
        assert probs[cell] == pytest.approx(1 / 5, abs=EPS), f"{cell}: {probs[cell]}"


def test_flagged_mine_reduces_remaining_mines():
    # 3x3 with total mines = 2. One flagged.
    # View:
    #   1 . .
    #   F . .
    #   . . .
    # "1" sees flagged (1,0): effective n_mines = 0 for {(0,1),(1,1)}.
    # So both are safe. Remaining mine total = 1, off-frontier cells =
    # everything else unrevealed = {(0,2),(1,2),(2,0),(2,1),(2,2)} = 5 cells.
    # The two frontier cells become part of off-frontier once they're known
    # safe -- but ProbabilityAgent doesn't auto-merge, so it should still
    # report P=0 for them.
    view = np.array([
        [1, U, U],
        [F, U, U],
        [U, U, U],
    ], dtype=np.int8)
    agent = make_agent(3, 3, 2)
    probs = all_probs(agent, view)

    assert probs[(0, 1)] == pytest.approx(0.0, abs=EPS)
    assert probs[(1, 1)] == pytest.approx(0.0, abs=EPS)

    # Remaining 1 mine across 5 off-frontier cells -> 1/5 each.
    off = {(0, 2), (1, 2), (2, 0), (2, 1), (2, 2)}
    for cell in off:
        assert probs[cell] == pytest.approx(1 / 5, abs=EPS), f"{cell}: {probs[cell]}"


def test_two_disjoint_components_independent():
    # Two separate "1" constraints, no shared cells, exactly 2 mines total.
    # Both components have 1 mine each (forced by total). Each cell in each
    # component has P = 1/k where k is that component's size.
    #
    # Layout 3x5:
    #   1 . . . 1
    #   . . . . .
    #   . . . . .
    # (0,0) "1" has unrevealed neighbors {(0,1),(1,0),(1,1)} -> 1 mine.
    # (0,4) "1" has unrevealed neighbors {(0,3),(1,3),(1,4)} -> 1 mine.
    # Total mines = 2 -> off-frontier 0 mines.
    view = np.array([
        [1, U, U, U, 1],
        [U, U, U, U, U],
        [U, U, U, U, U],
    ], dtype=np.int8)
    agent = make_agent(3, 5, 2)
    probs = all_probs(agent, view)

    left = {(0, 1), (1, 0), (1, 1)}
    right = {(0, 3), (1, 3), (1, 4)}
    for cell in left | right:
        assert probs[cell] == pytest.approx(1 / 3, abs=EPS), f"{cell}: {probs[cell]}"

    off = {(1, 2), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4)}
    for cell in off:
        assert probs[cell] == pytest.approx(0.0, abs=EPS), f"{cell}: {probs[cell]}"


def test_pure_offfrontier_uniform_when_no_constraints():
    # View with no numeric clues -> no constraints, no frontier. Every cell is
    # off-frontier and should get P(mine) = n_mines / n_cells.
    view = np.full((3, 3), U, dtype=np.int8)
    agent = make_agent(3, 3, 1)
    probs = all_probs(agent, view)
    # No components -> _compute_probabilities still assigns off-frontier prob.
    expected = 1 / 9
    for r in range(3):
        for c in range(3):
            assert probs[(r, c)] == pytest.approx(expected, abs=EPS), \
                f"{(r, c)}: {probs[(r, c)]}"


def test_act_first_move_is_center():
    view = np.full((5, 5), U, dtype=np.int8)
    agent = make_agent(5, 5, 5)
    action = agent.act(view)
    assert action == ("reveal", 2, 2)
