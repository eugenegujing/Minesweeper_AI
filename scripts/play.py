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
    p.add_argument("--save-replay", type=str, default=None,
                   help="Save a GIF replay of the game to this path")
    args = p.parse_args()

    h, w, m = DIFFICULTIES[args.difficulty]
    env = MinesweeperEnv(height=h, width=w, n_mines=m)
    agent_cls = AGENT_REGISTRY[args.agent]
    agent = agent_cls(h, w, m, rng=np.random.default_rng(args.seed))

    recorder = None
    if args.save_replay:
        from visualization import ReplayRecorder
        recorder = ReplayRecorder(h, w, m)

    obs, _ = env.reset(seed=args.seed)
    agent.reset()
    if recorder:
        recorder.snapshot(obs, None, None, 0)
    step = 0
    while True:
        action = agent.act(obs)
        idx = Agent.to_action_index(action, env.w, env.h)
        obs, reward, terminated, truncated, info = env.step(idx)
        step += 1
        if recorder:
            probs = getattr(agent, "last_probabilities", None) or None
            reason = getattr(agent, "last_reason", "")
            recorder.snapshot(obs, probs, action, step, title_extra=reason)
        if args.render:
            reason = getattr(agent, "last_reason", "")
            suffix = f"  [{reason}]" if reason else ""
            print(f"--- step {step}: {action} reward={reward:+.3f}{suffix}")
            print(env.render())
        if terminated or truncated:
            outcome = "WIN" if info["won"] else ("LOSS" if info["lost"] else "TRUNC")
            print(f"\n{outcome} after {step} steps "
                  f"({h}x{w}, {m} mines, agent={args.agent})")
            if recorder:
                recorder.save_gif(args.save_replay)
                print(f"Saved replay to {args.save_replay}")
            return


if __name__ == "__main__":
    main()
