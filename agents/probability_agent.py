"""Probability-based agent.

Skeleton implementation. The plan:

1. Build the constraint graph (same as CSPAgent._build_constraints).
2. Reduce constraints by certain mines/safe cells found by the CSP layer.
3. Split frontier variables into connected components (cells share a component if
   they appear in a common constraint).
4. For each component, enumerate all assignments consistent with its constraints
   (DFS with pruning). Weight each assignment by C(remaining_off_frontier_cells,
   remaining_mines - mines_used). Aggregate marginal P(mine) per cell.
5. For off-frontier interior cells, use the global density adjusted by the
   distribution over mines used in the frontier.
6. Pick the cell with minimum P(mine). Break ties by preferring frontier cells
   (which yield more information when revealed safely).

For now this is a thin wrapper that defers to a random fallback when no certain
action is available.
"""
from __future__ import annotations

import numpy as np

from .csp_agent import CSPAgent
from .base import Action


class ProbabilityAgent(CSPAgent):
    name = "probability"

    def act(self, view: np.ndarray) -> Action:
        # TODO: replace _fallback with frontier enumeration. Until that lands,
        # we behave like CSPAgent so the harness end-to-end tests pass.
        return super().act(view)
