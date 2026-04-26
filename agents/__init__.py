from .base import Agent
from .random_agent import RandomAgent
from .csp_agent import CSPAgent
from .probability_agent import ProbabilityAgent
from .hybrid_agent import HybridAgent

AGENT_REGISTRY = {
    "random": RandomAgent,
    "csp": CSPAgent,
    "probability": ProbabilityAgent,
    "hybrid": HybridAgent,
}

__all__ = ["Agent", "RandomAgent", "CSPAgent", "ProbabilityAgent",
           "HybridAgent", "AGENT_REGISTRY"]
