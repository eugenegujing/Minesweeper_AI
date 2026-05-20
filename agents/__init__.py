from .base import Agent
from .random_agent import RandomAgent
from .single_point_agent import SinglePointAgent
from .csp_agent import CSPAgent
from .probability_agent import ProbabilityAgent
from .final_agent import FinalAgent
from .dqn_agent import DQNAgent

AGENT_REGISTRY = {
    "random": RandomAgent,
    "single_point": SinglePointAgent,
    "csp": CSPAgent,
    "probability": ProbabilityAgent,
    "final": FinalAgent,
    "dqn": DQNAgent,
}

__all__ = ["Agent", "RandomAgent", "SinglePointAgent", "CSPAgent",
           "ProbabilityAgent", "FinalAgent", "DQNAgent", "AGENT_REGISTRY"]
