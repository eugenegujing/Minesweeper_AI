"""Hybrid agent: CSP first, probability when stuck.

For now this is an alias for ProbabilityAgent (since ProbabilityAgent already
calls into CSP via its parent). Kept as a separate class so we can swap in a
learned probability head later without changing scripts.
"""
from .probability_agent import ProbabilityAgent


class HybridAgent(ProbabilityAgent):
    name = "hybrid"
