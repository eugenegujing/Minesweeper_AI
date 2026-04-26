from .base import Agent
from .random_agent import RandomAgent
from .csp_agent import CSPAgent
from .probability_agent import ProbabilityAgent
from .final_agent import FinalAgent

AGENT_REGISTRY = {
    "random": RandomAgent,
    "csp": CSPAgent,
    "probability": ProbabilityAgent,
    "final": FinalAgent,
}

__all__ = ["Agent", "RandomAgent", "CSPAgent", "ProbabilityAgent",
           "FinalAgent", "AGENT_REGISTRY"]
