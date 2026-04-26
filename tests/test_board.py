"""Smoke tests for the board logic."""
from __future__ import annotations

import numpy as np

from minesweeper.board import Board, CellState


def test_first_click_is_safe():
    rng = np.random.default_rng(0)
    for _ in range(20):
        b = Board(9, 9, 10, rng=rng)
        ok = b.reveal(4, 4)
        assert ok, "first click must always be safe"
        assert not b.lost


def test_full_reveal_wins():
    rng = np.random.default_rng(1)
    b = Board(5, 5, 3, rng=rng)
    b.reveal(0, 0)  # places mines, opens region
    # Reveal every non-mine cell.
    for r in range(b.h):
        for c in range(b.w):
            if not b._mines[r, c]:
                b.reveal(r, c)
    assert b.won
    assert not b.lost


def test_view_codes():
    b = Board(3, 3, 1, rng=np.random.default_rng(2))
    v = b.view()
    assert (v == int(CellState.UNREVEALED)).all()
    b.toggle_flag(0, 0)
    assert b.view()[0, 0] == int(CellState.FLAGGED)
