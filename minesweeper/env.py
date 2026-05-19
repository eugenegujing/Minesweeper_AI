"""Gymnasium-compatible Minesweeper environment.

Action space is Discrete(2 * H * W):
  - actions in [0, H*W)        -> reveal cell at index a
  - actions in [H*W, 2*H*W)    -> toggle flag at index (a - H*W)

Observation is the int8 board view from `Board.view()`.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    _HAS_GYM = True
except ImportError:  # pragma: no cover - gymnasium is optional for unit tests
    _HAS_GYM = False
    gym = object  # type: ignore

from .board import Board, DIFFICULTIES


class MinesweeperEnv(gym.Env if _HAS_GYM else object):
    metadata = {"render_modes": ["ansi", "human"]}

    REWARD_WIN = 1.0
    REWARD_LOSS = -1.0
    REWARD_REVEAL = 0.01     # tiny dense reward for safe reveal
    REWARD_INVALID = -0.001  # discourage clicking already-revealed cells

    def __init__(self, height: int = 9, width: int = 9, n_mines: int = 10,
                 difficulty: Optional[str] = None,
                 max_steps: Optional[int] = None,
                 render_mode: Optional[str] = None):
        if difficulty is not None:
            height, width, n_mines = DIFFICULTIES[difficulty]
        self.h, self.w, self.n_mines = height, width, n_mines
        self.max_steps = max_steps if max_steps is not None else 4 * height * width
        self.render_mode = render_mode
        self._steps = 0
        self.board: Optional[Board] = None

        if _HAS_GYM:
            self.action_space = spaces.Discrete(2 * height * width)
            self.observation_space = spaces.Box(
                low=-3, high=8, shape=(height, width), dtype=np.int8
            )

    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if _HAS_GYM:
            super().reset(seed=seed)
        rng = np.random.default_rng(seed)
        self.board = Board(self.h, self.w, self.n_mines, rng=rng)
        self._steps = 0
        return self.board.view(), {}

    def step(self, action: int):
        assert self.board is not None, "call reset() first"
        self._steps += 1
        n = self.h * self.w
        if action < n:
            r, c = divmod(action, self.w)
            already_revealed = bool(self.board._revealed[r, c])
            safe = self.board.reveal(r, c)
            if not safe:
                reward = self.REWARD_LOSS
            elif self.board.won:
                reward = self.REWARD_WIN
            elif already_revealed:
                reward = self.REWARD_INVALID
            else:
                reward = self.REWARD_REVEAL
        else:
            r, c = divmod(action - n, self.w)
            self.board.toggle_flag(r, c)
            reward = 0.0
        terminated = self.board.lost or self.board.won
        truncated = self._steps >= self.max_steps and not terminated
        info = {"won": self.board.won, "lost": self.board.lost,
                "n_remaining": self.board.n_remaining,
                "n_revealed_safe": self.board.n_revealed_safe}
        return self.board.view(), reward, terminated, truncated, info

    def render(self):
        if self.board is None:
            return ""
        return self.board.render_ascii()
