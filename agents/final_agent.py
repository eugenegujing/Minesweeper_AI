"""Final agent. TODO: fill in.

Intended to be the project's strongest agent — likely a combination of CSP
reasoning, probability estimation, and possibly a learned component. Left
empty for now so it can be designed once the underlying pieces are in place.
"""
from __future__ import annotations

from .base import Action, Agent


class FinalAgent(Agent):
    name = "final"

    def act(self, view):  # type: ignore[override]
        raise NotImplementedError("FinalAgent.act is not implemented yet")
