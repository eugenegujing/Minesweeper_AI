"""Render-and-replay helpers for Minesweeper game visualization.

Each step of a game can be rendered as a PNG showing:
- The board state (revealed numbers, flags, unrevealed cells)
- Mine probabilities as a heat overlay (when ProbabilityAgent provided them)
- A highlight on the cell of the most recent action

A series of PNGs is then assembled into a GIF replay.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from typing import Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from minesweeper.board import CellState


Cell = tuple[int, int]
Action = tuple[str, int, int]

NUMBER_COLORS = {
    1: "#0000ff", 2: "#008000", 3: "#ff0000", 4: "#000080",
    5: "#800000", 6: "#008080", 7: "#000000", 8: "#808080",
}


def render_step(
    view: np.ndarray,
    probabilities: Optional[dict[Cell, float]],
    action: Optional[Action],
    step: int,
    save_path: str,
    title_extra: str = "",
) -> None:
    h, w = view.shape
    fig, ax = plt.subplots(figsize=(w * 0.4 + 1, h * 0.4 + 1.2))

    cmap = plt.cm.RdYlGn_r
    UNREVEALED = int(CellState.UNREVEALED)
    FLAGGED = int(CellState.FLAGGED)
    MINE = int(CellState.MINE)

    for r in range(h):
        for c in range(w):
            v = int(view[r, c])
            y = h - 1 - r  # flip so row 0 appears at top

            if v == UNREVEALED:
                if probabilities and (r, c) in probabilities:
                    p = probabilities[(r, c)]
                    facecolor = cmap(p)
                    ax.add_patch(patches.Rectangle(
                        (c, y), 1, 1, facecolor=facecolor,
                        edgecolor="gray", linewidth=0.5))
                    ax.text(c + 0.5, y + 0.5, f"{int(round(p * 100))}",
                            ha="center", va="center", fontsize=6, color="black")
                else:
                    ax.add_patch(patches.Rectangle(
                        (c, y), 1, 1, facecolor="#cccccc",
                        edgecolor="gray", linewidth=0.5))
            elif v == FLAGGED:
                ax.add_patch(patches.Rectangle(
                    (c, y), 1, 1, facecolor="#ff8c42",
                    edgecolor="gray", linewidth=0.5))
                ax.text(c + 0.5, y + 0.5, "F",
                        ha="center", va="center", fontsize=10,
                        fontweight="bold", color="white")
            elif v == MINE:
                ax.add_patch(patches.Rectangle(
                    (c, y), 1, 1, facecolor="red",
                    edgecolor="black", linewidth=1))
                ax.text(c + 0.5, y + 0.5, "*",
                        ha="center", va="center", fontsize=14,
                        fontweight="bold", color="white")
            else:
                ax.add_patch(patches.Rectangle(
                    (c, y), 1, 1, facecolor="white",
                    edgecolor="gray", linewidth=0.5))
                if v > 0:
                    ax.text(c + 0.5, y + 0.5, str(v),
                            ha="center", va="center", fontsize=10,
                            fontweight="bold",
                            color=NUMBER_COLORS.get(v, "black"))

    if action is not None:
        kind, ar, ac = action
        edge = "#0066cc" if kind == "reveal" else "#ff8c42"
        ax.add_patch(patches.Rectangle(
            (ac, h - 1 - ar), 1, 1, facecolor="none",
            edgecolor=edge, linewidth=2.5))

    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")

    if action is not None:
        title = f"Step {step}: {action[0]} ({action[1]}, {action[2]})"
    else:
        title = f"Step {step}"
    if title_extra:
        title = f"{title} — {title_extra}"
    ax.set_title(title, fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=80, bbox_inches="tight")
    plt.close(fig)


class ReplayRecorder:
    def __init__(self, height: int, width: int, n_mines: int):
        self.h = height
        self.w = width
        self.n_mines = n_mines
        self.tmp_dir = tempfile.mkdtemp(prefix="msweeper_replay_")
        self.frames: list[str] = []

    def snapshot(
        self,
        view: np.ndarray,
        probabilities: Optional[dict[Cell, float]],
        action: Optional[Action],
        step: int,
        title_extra: str = "",
    ) -> None:
        png_path = os.path.join(self.tmp_dir, f"step_{step:04d}.png")
        render_step(view, probabilities, action, step, png_path, title_extra)
        self.frames.append(png_path)

    def save_gif(self, output_path: str, fps: float = 2.0) -> None:
        if not self.frames:
            return
        images = [Image.open(p).convert("RGB") for p in self.frames]
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=int(1000 / fps),
            loop=0,
        )
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
