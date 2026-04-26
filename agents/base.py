"""Abstract agent interface.

An Agent observes the int8 board view and emits one of:
  - ("reveal", r, c)
  - ("flag",   r, c)
The runner translates this into the env's flat Discrete action.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

Action = Tuple[str, int, int]  # (kind, row, col); kind in {"reveal", "flag"}


class Agent(ABC):
    name: str = "base"

    def __init__(self, height: int, width: int, n_mines: int,
                 rng: np.random.Generator | None = None):
        self.h = height
        self.w = width
        self.n_mines = n_mines
        self.rng = rng if rng is not None else np.random.default_rng()

    def reset(self) -> None:
        """Called at the start of each episode. Override to clear internal state."""

    @abstractmethod
    def act(self, view: np.ndarray) -> Action:
        ...

    # ----- helpers shared by subclasses -----
    @staticmethod
    def unrevealed_cells(view: np.ndarray) -> list[tuple[int, int]]:
        from minesweeper.board import CellState
        coords = np.argwhere(view == int(CellState.UNREVEALED))
        return [(int(r), int(c)) for r, c in coords]

    @staticmethod
    def to_action_index(action: Action, width: int, height: int) -> int:
        kind, r, c = action
        flat = r * width + c
        if kind == "reveal":
            return flat
        if kind == "flag":
            return height * width + flat
        raise ValueError(f"unknown action kind: {kind}")
