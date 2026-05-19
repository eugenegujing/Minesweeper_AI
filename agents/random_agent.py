"""Random baseline agent: picks any unrevealed, unflagged cell."""
from __future__ import annotations

import numpy as np

from .base import Action, Agent


class RandomAgent(Agent):
    name = "random"

    def act(self, view: np.ndarray) -> Action:
        self.last_was_guess = True
        self.last_reason = "random"
        cells = self.unrevealed_cells(view)
        if not cells:
            return ("reveal", 0, 0)
        idx = int(self.rng.integers(len(cells)))
        r, c = cells[idx]
        return ("reveal", r, c)
