"""Deep Q-Network agent (pure, reveal-only).

The network takes the 11-channel one-hot encoded board view and outputs one
Q-value per cell, interpreted as the value of revealing that cell. At inference
time we mask out cells that are already revealed (a no-op in the env) and
argmax over the remainder. No classical reasoning is mixed in.

A trained checkpoint must be placed at the path passed via `checkpoint_path`
(or the project-default `checkpoints/dqn_<difficulty>.pt`). If no checkpoint
is found, the agent emits warnings and falls back to a random unrevealed cell,
so the rest of the test/benchmark harness keeps working.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from minesweeper.board import CellState

from .base import Action, Agent


# Channel ordering used by both training and inference. The agent never sees
# MINE (-3) during act() because the game has ended, so we don't reserve a
# channel for it.
_CHANNEL_VALUES = (
    int(CellState.FLAGGED),    # -2
    int(CellState.UNREVEALED), # -1
    0, 1, 2, 3, 4, 5, 6, 7, 8,
)
N_CHANNELS = len(_CHANNEL_VALUES)  # 11


def encode_view(view: np.ndarray) -> np.ndarray:
    """Convert an (H, W) int8 board view to an (N_CHANNELS, H, W) float32 array."""
    h, w = view.shape
    out = np.zeros((N_CHANNELS, h, w), dtype=np.float32)
    for i, val in enumerate(_CHANNEL_VALUES):
        out[i] = (view == val).astype(np.float32)
    return out


class QNetwork(nn.Module):
    """Small fully-convolutional Q-network. Output shape matches the board."""

    def __init__(self, n_channels: int = N_CHANNELS, hidden: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1)
        self.head = nn.Conv2d(hidden, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.head(x)
        return x.squeeze(1)  # (B, H, W)


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def default_checkpoint_path(height: int, width: int, n_mines: int) -> Path:
    from minesweeper.board import DIFFICULTIES
    tag = f"{height}x{width}_{n_mines}"
    for name, (h, w, m) in DIFFICULTIES.items():
        if (h, w, m) == (height, width, n_mines):
            tag = name
            break
    root = Path(__file__).resolve().parent.parent
    return root / "checkpoints" / f"dqn_{tag}.pt"


class DQNAgent(Agent):
    name = "dqn"

    def __init__(self, height: int, width: int, n_mines: int,
                 rng: Optional[np.random.Generator] = None,
                 checkpoint_path: Optional[str | Path] = None,
                 device: Optional[torch.device] = None):
        super().__init__(height, width, n_mines, rng=rng)
        self.device = device if device is not None else default_device()
        self.net = QNetwork().to(self.device)
        self.net.eval()
        self._loaded = False
        path = Path(checkpoint_path) if checkpoint_path \
            else default_checkpoint_path(height, width, n_mines)
        self.checkpoint_path = path
        if path.exists():
            state = torch.load(path, map_location=self.device, weights_only=True)
            self.net.load_state_dict(state["model"] if "model" in state else state)
            self._loaded = True
        else:
            warnings.warn(
                f"DQN checkpoint not found at {path}; agent will play randomly. "
                f"Train one with `python -m scripts.train_dqn`.",
                stacklevel=2,
            )

    @torch.no_grad()
    def act(self, view: np.ndarray) -> Action:
        if not self._loaded:
            return self._random_fallback(view, reason="no checkpoint loaded")

        enc = encode_view(view)
        x = torch.from_numpy(enc).unsqueeze(0).to(self.device)
        q = self.net(x).squeeze(0).cpu().numpy()  # (H, W)

        # Mask non-unrevealed cells (revealed or flagged) to -inf.
        unrevealed_mask = (view == int(CellState.UNREVEALED))
        if not unrevealed_mask.any():
            return ("reveal", 0, 0)
        q_masked = np.where(unrevealed_mask, q, -np.inf)

        flat = int(np.argmax(q_masked))
        r, c = divmod(flat, self.w)
        self.last_was_guess = True  # DQN never *proves* a cell is safe
        self.last_reason = f"DQN argmax (Q={float(q[r, c]):.3f})"
        return ("reveal", r, c)

    def _random_fallback(self, view: np.ndarray, reason: str) -> Action:
        cells = self.unrevealed_cells(view)
        if not cells:
            return ("reveal", 0, 0)
        self.last_was_guess = True
        self.last_reason = f"random fallback ({reason})"
        idx = int(self.rng.integers(len(cells)))
        r, c = cells[idx]
        return ("reveal", r, c)
