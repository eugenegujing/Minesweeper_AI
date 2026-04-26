"""Play a single Minesweeper game with a chosen agent.

Usage:
    python -m scripts.play --agent csp --difficulty beginner --render
"""
from __future__ import annotations

import argparse

import numpy as np

from agents import AGENT_REGISTRY
from agents.base import Agent
from minesweeper.board import DIFFICULTIES
from minesweeper.env import MinesweeperEnv


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--agent", choices=sorted(AGENT_REGISTRY), default="random")
    p.add_argument("--difficulty", choices=sorted(DIFFICULTIES), default="beginner")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--render", action="store_true")
    args = p.parse_args()

    h, w, m = DIFFICULTIES[args.difficulty]
    env = MinesweeperEnv(height=h, width=w, n_mines=m)
    agent_cls = AGENT_REGISTRY[args.agent]
    agent = agent_cls(h, w, m, rng=np.random.default_rng(args.seed))

    obs, _ = env.reset(seed=args.seed)
    agent.reset()
    step = 0
    while True:
        action = agent.act(obs)
        idx = Agent.to_action_index(action, env.w, env.h)
        obs, reward, terminated, truncated, info = env.step(idx)
        step += 1
        if args.render:
            print(f"--- step {step}: {action} reward={reward:+.3f}")
            print(env.render())
        if terminated or truncated:
            outcome = "WIN" if info["won"] else ("LOSS" if info["lost"] else "TRUNC")
            print(f"\n{outcome} after {step} steps "
                  f"({h}x{w}, {m} mines, agent={args.agent})")
            return


if __name__ == "__main__":
    main()
