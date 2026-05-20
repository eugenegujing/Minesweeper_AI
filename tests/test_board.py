"""Unit tests for core board mechanics."""
from __future__ import annotations

import numpy as np

from minesweeper.board import Board, CellState


def test_first_click_is_safe_and_protects_neighbors():
    board = Board(9, 9, 10, rng=np.random.default_rng(0))

    assert board.reveal(4, 4) is True
    assert board.first_action == (4, 4)
    assert not board._mines[4, 4]

    for nr, nc in board._neighbors(4, 4):
        assert not board._mines[nr, nc]


def test_zero_mine_board_flood_reveals_everything_and_wins():
    board = Board(3, 3, 0, rng=np.random.default_rng(0))

    assert board.reveal(1, 1) is True
    assert board.won is True
    assert board.n_remaining == 0
    assert (board.view() >= 0).all()


def test_flagged_cell_is_not_revealed_or_used_as_first_click():
    board = Board(3, 3, 1, rng=np.random.default_rng(0))

    board.toggle_flag(1, 1)
    assert board.reveal(1, 1) is True

    view = board.view()
    assert view[1, 1] == int(CellState.FLAGGED)
    assert board._mines_placed is False


def test_revealing_mine_loses_and_view_marks_mine():
    board = Board(2, 2, 1, rng=np.random.default_rng(0), safe_first_click=False)
    board._mines[0, 0] = True
    board._counts[:, :] = 1
    board._counts[0, 0] = -1
    board._mines_placed = True

    assert board.reveal(0, 0) is False
    assert board.lost is True
    assert board.view()[0, 0] == int(CellState.MINE)


def test_win_condition_after_all_safe_cells_revealed():
    board = Board(2, 2, 1, rng=np.random.default_rng(0), safe_first_click=False)
    board._mines[0, 0] = True
    board._counts[:, :] = 1
    board._counts[0, 0] = -1
    board._mines_placed = True

    assert board.reveal(0, 1) is True
    assert board.reveal(1, 0) is True
    assert board.reveal(1, 1) is True
    assert board.won is True
